import json
import locale
from datetime import datetime
from itertools import chain
from typing import Dict, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sheets.utils import format_number_str, get_code_departement

from ..constants import BSDA, BSDASRI, BSDD, BSDD_NON_DANGEROUS, BSFF, BSVHU

# classes returning a serialized (json) plotly visualization to be consumed by a plotly script
locale.setlocale(locale.LC_ALL, "fr_FR")


class BsdQuantitiesGraph:
    """Component with a Line Figure showing incoming and outgoing quantities of waste.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bs_data: DataFrame
        DataFrame containing data for a given 'bordereau' type.
    quantity_variables_names: list of str
        The names of the variables to use to compute quantity statistics. Several variables can be used.
    packagings_data : DataFrame
        For BSFF data, packagings dataset to be able to compute the quantities.
    """

    def __init__(
        self,
        company_siret: str,
        bs_type: str,
        bs_data: pd.DataFrame,
        data_date_interval: tuple[datetime, datetime],
        quantity_variables_names: list[str] = ["quantity_received"],
        packagings_data: pd.DataFrame | None = None,
    ):
        self.bs_data = bs_data
        self.bs_type = bs_type
        self.packagings_data = packagings_data
        self.company_siret = company_siret
        self.data_date_interval = data_date_interval
        self.quantity_variables_names = self._validate_quantity_variables_names(
            quantity_variables_names, packagings_data
        )

        self.incoming_data_by_month_series = []
        self.outgoing_data_by_month_series = []

        self.figure = None

    @staticmethod
    def _validate_quantity_variables_names(quantity_variables_names, packagings_data):
        allowed_quantity_variables_names = [
            "quantity_received",
            "acceptation_weight",
            "volume",
        ]

        clean_quantity_variables_names = [e for e in quantity_variables_names if e in allowed_quantity_variables_names]

        # Allows to handle the case when there is no packagings data but there is BSFF data
        if packagings_data is None:
            clean_quantity_variables_names = [
                e if e != "acceptation_weight" else "quantity_received" for e in clean_quantity_variables_names
            ]

        return list(set(clean_quantity_variables_names))

    def _preprocess_data(self) -> None:
        bs_data = self.bs_data.copy()

        incoming_data = bs_data[
            (bs_data["recipient_company_siret"] == self.company_siret)
            & bs_data["received_at"].between(*self.data_date_interval)
        ]

        outgoing_data = bs_data[
            (bs_data["emitter_company_siret"] == self.company_siret)
            & bs_data["sent_at"].between(*self.data_date_interval)
        ]

        # We iterate over the different variables chosen to compute the statistics
        for variable_name in self.quantity_variables_names:
            # If there is a packagings_data DataFrame, then it means that we are
            # computing BSFF statistics, in this case we use the packagings data instead of
            # 'bordereaux' data as quantity information is stored at packaging level
            incoming_data = incoming_data.copy()
            outgoing_data = outgoing_data.copy()

            if self.bs_type == BSFF:
                if self.packagings_data is None:
                    # Case when there is BSFFs but no packagings info
                    continue
                incoming_data_by_month = (
                    incoming_data.merge(self.packagings_data, left_on="id", right_on="bsff_id")
                    .groupby(pd.Grouper(key="acceptation_date", freq="1M"))[variable_name]
                    .sum()
                    .replace(0, np.nan)
                )
                outgoing_data_by_month = (
                    outgoing_data.merge(self.packagings_data, left_on="id", right_on="bsff_id")
                    .groupby(pd.Grouper(key="sent_at", freq="1M"))[variable_name]
                    .sum()
                    .replace(0, np.nan)
                )
            else:
                # Handle quantity refused
                if self.bs_type in [BSDD, BSDD_NON_DANGEROUS, BSDASRI]:
                    incoming_data["quantity_received"] = incoming_data["quantity_received"] - incoming_data[
                        "quantity_refused"
                    ].fillna(0)

                    outgoing_data["quantity_received"] = outgoing_data["quantity_received"] - outgoing_data[
                        "quantity_refused"
                    ].fillna(0)

                outgoing_data_by_month = (
                    outgoing_data.groupby(pd.Grouper(key="sent_at", freq="1M"))[variable_name].sum().replace(0, np.nan)
                )

                incoming_data_by_month = (
                    incoming_data.groupby(pd.Grouper(key="received_at", freq="1M"))[variable_name]
                    .sum()
                    .replace(0, np.nan)
                )

            self.incoming_data_by_month_series.append(incoming_data_by_month)
            self.outgoing_data_by_month_series.append(outgoing_data_by_month)

    def _check_data_empty(self) -> bool:
        incoming_data_by_month_series = self.incoming_data_by_month_series
        outgoing_data_by_month_series = self.outgoing_data_by_month_series

        # If DataFrames are empty then output is empty
        if all(len(s) == len(z) == 0 for s, z in zip(incoming_data_by_month_series, outgoing_data_by_month_series)):
            return True

        # If preprocessed series are full of NA then output is empty
        if all(
            s.isna().all() and z.isna().all()
            for s, z in zip(incoming_data_by_month_series, outgoing_data_by_month_series)
        ):
            return True

        # If preprocessed series are full of 0's then output is empty
        if all(
            (s == 0).all() and (z == 0).all()
            for s, z in zip(incoming_data_by_month_series, outgoing_data_by_month_series)
        ):
            return True

        return False

    def _create_figure(self) -> None:
        fig = go.Figure()

        lines = []  # Will store the lines graph objects

        # We store the minimum date of each series to be able to configure
        # the tick 0 of the figure
        mins_x = []

        # This is used to configure the dticks in case of low number of data points.
        numbers_of_data_points = []

        # We create two lines (for incoming and outgoing) for each quantity variable chosen
        for variable_name, incoming_data_by_month, outgoing_data_by_month in zip(
            self.quantity_variables_names,
            self.incoming_data_by_month_series,
            self.outgoing_data_by_month_series,
        ):
            incoming_line_name = "Quantité entrante"
            incoming_hover_text = "{} - <b>{}</b> tonnes entrantes"
            outgoing_line_name = "Quantité sortante"
            outgoing_hover_text = "{} - <b>{}</b> tonnes sortantes"
            marker_line_style = "solid"
            marker_symbol = "circle"
            marker_size = 6

            # To handle the case of volume
            if variable_name == "volume":
                incoming_line_name = "Volume entrant"
                incoming_hover_text = "{} - <b>{}</b> m³ entrants"
                outgoing_line_name = "Volume sortant"
                outgoing_hover_text = "{} - <b>{}</b> m³ sortants"
                marker_line_style = "dash"
                marker_symbol = "triangle-up"
                marker_size = 10

            if len(incoming_data_by_month) > 0:
                incoming_line = go.Scatter(
                    x=incoming_data_by_month.index,
                    y=incoming_data_by_month,
                    name=incoming_line_name,
                    mode="lines+markers",
                    hovertext=[
                        incoming_hover_text.format(index.strftime("%B %y").capitalize(), format_number_str(e))
                        for index, e in incoming_data_by_month.items()
                    ],
                    hoverinfo="text",
                    marker_color="#E1000F",
                    marker_symbol=marker_symbol,
                    marker_size=marker_size,
                    line_dash=marker_line_style,
                )
                mins_x.append(incoming_data_by_month.index.min())
                numbers_of_data_points.append(len(incoming_data_by_month))
                lines.append(incoming_line)

            if len(outgoing_data_by_month) > 0:
                outgoing_line = go.Scatter(
                    x=outgoing_data_by_month.index,
                    y=outgoing_data_by_month,
                    name=outgoing_line_name,
                    mode="lines+markers",
                    hovertext=[
                        outgoing_hover_text.format(index.strftime("%B %y").capitalize(), format_number_str(e))
                        for index, e in outgoing_data_by_month.items()
                    ],
                    hoverinfo="text",
                    marker_color="#6A6AF4",
                    marker_symbol=marker_symbol,
                    marker_size=marker_size,
                    line_dash=marker_line_style,
                )
                mins_x.append(outgoing_data_by_month.index.min())
                numbers_of_data_points.append(len(outgoing_data_by_month))
                lines.append(outgoing_line)

        fig.add_traces(lines)

        dtick = "M2"
        if not numbers_of_data_points or max(numbers_of_data_points) < 3:
            dtick = "M1"

        tickangle = 0
        y_legend = -0.07
        if numbers_of_data_points and max(numbers_of_data_points) >= 15:
            tickangle = -90
            y_legend = -0.12

        fig.update_layout(
            margin={"t": 20, "l": 35, "r": 5},
            legend={"orientation": "h", "y": y_legend, "x": 0},
            legend_font_size=11,
            legend_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        fig.update_xaxes(
            tickangle=tickangle,
            tickformat="%b %y",
            tick0=min(mins_x) if mins_x else None,
            dtick=dtick,
            gridcolor="#ccc",
        )
        fig.update_yaxes(exponentformat="B", tickformat=".2s", gridcolor="#ccc")

        self.figure = fig

    def build(self):
        self._preprocess_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class BsdTrackedAndRevisedProcessor:
    """Component with a Bar Figure of emitted, received and revised 'bordereaux'.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bs_data: DataFrame
        DataFrame containing data for a given 'bordereau' type.
    data_date_interval: tuple
        Date interval to filter data.
    bs_revised_data: DataFrame
        Optional DataFrame containing list of revised 'bordereaux' for a given 'bordereau' type.
    """

    def __init__(
        self,
        company_siret: str,
        bs_data: pd.DataFrame,
        data_date_interval: tuple[datetime, datetime],
        bs_revised_data: pd.DataFrame | None = None,
    ) -> None:
        self.company_siret = company_siret
        self.bs_data = bs_data
        self.data_date_interval = data_date_interval
        self.bs_revised_data = bs_revised_data
        self.bs_emitted_by_month = None
        self.bs_received_by_month = None
        self.bs_revised_by_month = None

        self.figure = None

    def _preprocess_bs_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it for plotting."""
        bs_data = self.bs_data

        bs_emitted = bs_data[
            (bs_data["emitter_company_siret"] == self.company_siret)
            & bs_data["sent_at"].between(*self.data_date_interval)
        ].dropna(subset=["sent_at"])

        bs_emitted_by_month = bs_emitted.groupby(pd.Grouper(key="sent_at", freq="1M")).id.count()

        bs_received = bs_data[
            (bs_data["recipient_company_siret"] == self.company_siret)
            & bs_data["received_at"].between(*self.data_date_interval)
        ].dropna(subset=["received_at"])

        bs_received_by_month = bs_received.groupby(pd.Grouper(key="received_at", freq="1M")).id.count()

        self.bs_emitted_by_month = bs_emitted_by_month
        self.bs_received_by_month = bs_received_by_month

    def _preprocess_bs_revised_data(self) -> None:
        """Preprocess raw revised 'bordereaux' data to prepare it for plotting."""
        bs_revised_data = self.bs_revised_data

        bs_revised_data = bs_revised_data[
            bs_revised_data["bs_id"].isin(self.bs_data["id"])
            & (bs_revised_data["created_at"].between(*self.data_date_interval))
        ]
        bs_revised_by_month = bs_revised_data.groupby(pd.Grouper(key="created_at", freq="1M")).bs_id.nunique()

        self.bs_revised_by_month = bs_revised_by_month

    def _check_data_empty(self) -> bool:
        bs_emitted_by_month = self.bs_emitted_by_month
        bs_received_by_month = self.bs_received_by_month

        if len(bs_emitted_by_month) == len(bs_received_by_month) == 0:
            return True

        return False

    def _create_figure(self) -> None:
        bs_emitted_by_month = self.bs_emitted_by_month
        bs_received_by_month = self.bs_received_by_month
        bs_revised_by_month = self.bs_revised_by_month

        text_size = 12

        bs_emitted_bars = go.Bar(
            x=bs_emitted_by_month.index,
            y=bs_emitted_by_month,
            name="Bordereaux émis",
            hovertext=[
                "{} - <b>{}</b> bordereau(x) émis".format(index.strftime("%B %y").capitalize(), e)
                for index, e in bs_emitted_by_month.items()
            ],
            hoverinfo="text",
            textfont_size=text_size,
            textposition="outside",
            constraintext="none",
            marker_color="#6A6AF4",
        )

        bs_received_bars = go.Bar(
            x=bs_received_by_month.index,
            y=bs_received_by_month,
            name="Bordereaux reçus",
            hovertext=[
                "{} - <b>{}</b> bordereau(x) reçus".format(index.strftime("%B %y").capitalize(), e)
                for index, e in bs_received_by_month.items()
            ],
            hoverinfo="text",
            textfont_size=text_size,
            textposition="outside",
            constraintext="none",
            marker_color="#E1000F",
        )

        if pd.isna(bs_emitted_by_month.index.min()):
            tick0_min = bs_received_by_month.index.min()
        elif pd.isna(bs_received_by_month.index.min()):
            tick0_min = bs_emitted_by_month.index.min()
        else:
            tick0_min = min(bs_emitted_by_month.index.min(), bs_received_by_month.index.min())

        # Used to store the maximum value of each line
        # to be able to configure the height of the plotting area of the figure.
        max_y = max(bs_emitted_by_month.max(), bs_received_by_month.max())

        fig = go.Figure([bs_emitted_bars, bs_received_bars])

        max_points = max(len(bs_emitted_by_month), len(bs_received_by_month))

        tickangle = 0
        y_legend = -0.07
        if max_points >= 15:
            tickangle = -90
            y_legend = -0.15

        if bs_revised_by_month is not None:
            fig.add_trace(
                go.Bar(
                    x=bs_revised_by_month.index,
                    y=bs_revised_by_month,
                    name="Bordereaux corrigés",
                    hovertext=[
                        "{} - <b>{}</b> bordereau(x) révisés".format(index.strftime("%B %y").capitalize(), e)
                        for index, e in bs_revised_by_month.items()
                    ],
                    hoverinfo="text",
                    textfont_size=text_size,
                    textposition="outside",
                    constraintext="none",
                    marker_color="#B7A73F",
                )
            )
            tick0_min = min(tick0_min, bs_revised_by_month.index.min())
            max_y = max(max_y, bs_revised_by_month.max())
            max_points = max(max_points, len(bs_revised_by_month))

        fig.update_layout(
            margin={"t": 20, "l": 35, "r": 5},
            legend={
                "orientation": "h",
                "y": y_legend,
                "x": -0.1,
            },
            legend_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        ticklabelstep = 2
        if max_points <= 3:
            ticklabelstep = 1

        fig.update_xaxes(
            dtick=f"M{ticklabelstep}",
            tickangle=tickangle,
            tickformat="%b %y",
            tick0=tick0_min,
            ticks="outside",
            gridcolor="#ccc",
        )

        # Range of the y axis is increased to increase the height of the plotting are of the figure
        fig.update_yaxes(range=[0, max_y * 1.1], gridcolor="#ccc")

        self.figure = fig

    def build(self):
        self._preprocess_bs_data()
        if self.bs_revised_data is not None:
            self._preprocess_bs_revised_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class WasteOriginProcessor:
    """Component with a bar figure representing the quantity of waste received by départements (only TOP 6).

    Parameters
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    departements_regions_df: DataFrame
        Static data about regions and départements with their codes.
    data_date_interval: tuple
        Date interval to filter data.
    """

    def __init__(
        self,
        company_siret: str,
        bs_data_dfs: Dict[str, pd.DataFrame],
        departements_regions_df: pd.DataFrame,
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.bs_data_dfs = bs_data_dfs
        self.departements_regions_df = departements_regions_df
        self.data_date_interval = data_date_interval

        self.preprocessed_serie = None
        self.figure = None

    def _preprocess_data(self) -> None:
        if len(self.bs_data_dfs) == 0:
            return

        concat_df = pd.concat(
            [df[df["received_at"].between(*self.data_date_interval)] for df in self.bs_data_dfs.values()]
        )

        # The postal code is extracted from the address field using a simple regex
        concat_df["cp"] = concat_df["emitter_company_address"].str.extract(r"([0-9]{5})", expand=False)
        concat_df["code_dep"] = concat_df["cp"].apply(get_code_departement)
        if concat_df["code_dep"].isna().all():
            return
        # 'Bordereau' data is merged with INSEE geographical data
        concat_df = pd.merge(
            concat_df,
            self.departements_regions_df,
            left_on="code_dep",
            right_on="DEP",
            how="left",
            validate="many_to_one",
        )

        # We create the column `cp_formatted` that will hold the two first digit
        # (three in the case of DOM/TOM) of the postal code
        concat_df.loc[~concat_df["code_dep"].isna(), "cp_formatted"] = (
            concat_df["LIBELLE_dep"] + " (" + concat_df["code_dep"] + ")"
        )

        # We handle the case of failed postal code extraction
        concat_df.loc[concat_df["code_dep"].isna(), "cp_formatted"] = "Origine inconnue"

        # Handle quantity refused
        if "quantity_refused" in concat_df.columns:
            concat_df["quantity_received"] = concat_df["quantity_received"] - concat_df["quantity_refused"].fillna(0)

        serie = (
            concat_df[concat_df["recipient_company_siret"] == self.company_siret]
            .groupby("cp_formatted")["quantity_received"]
            .sum()
        )

        serie.sort_values(ascending=False, inplace=True)

        # Only TOP 5 'départements' are kept
        final_serie = serie[:5]
        # Remaining 'départements' are summed and displayed as "others"
        final_serie["Autres origines"] = serie[5:].sum()
        final_serie = final_serie.astype(int)
        final_serie = final_serie.round(2)

        final_serie = final_serie[final_serie > 0]

        self.preprocessed_serie = final_serie

    def _check_data_empty(self) -> bool:
        if (
            (self.preprocessed_serie is None)
            or self.preprocessed_serie.isna().all()
            or len(self.preprocessed_serie) == 0
        ):
            return True

        return False

    def _create_figure(self) -> None:
        # Prepare order for horizontal bar chart, preserving "Autre origines" as bottom bar
        serie = pd.concat((self.preprocessed_serie[-1:], self.preprocessed_serie[-2::-1]))

        # The bar chart has invisible bar (at *_annot positions) that will hold the labels
        # Invisible bar is the bar with width 0 but with a label.
        y_cats = [tup_e for e in serie.index for tup_e in (e, e + "_annot")]
        values = [tup_e for _, e in serie.items() for tup_e in (e, 0)]
        texts = [
            tup_e
            for index, value in serie.items()
            for tup_e in (
                "",
                f"<b>{format_number_str(value, precision=2)}t</b> - {index}",
            )
        ]
        hovertexts = [
            tup_e
            for index, value in serie.items()
            for tup_e in (
                f"{index} - <b>{format_number_str(value, precision=2)}t</b> reçues",
                "",
            )
        ]
        bar_trace = go.Bar(
            x=values,
            y=y_cats,
            orientation="h",
            text=texts,
            textfont_size=20,
            textposition="outside",
            width=[tup_e for e in values for tup_e in (0.7, 1)],
            hovertext=hovertexts,
            hoverinfo="text",
        )

        fig = go.Figure([bar_trace])
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, type="category")
        fig.update_layout(
            margin={"t": 20, "b": 0, "l": 0, "r": 0},
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        self.figure = fig

    def build(self):
        self._preprocess_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class WasteOriginsMapProcessor:
    """Component with a bubble map figure representing the quantity of waste received by regions.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    departements_regions_df: DataFrame
        Static data about regions and départements with their codes.
    regions_geodata: GeoDataFrame
        GeoDataFrame including regions geometries.
    data_date_interval: tuple
        Date interval to filter data.
    """

    def __init__(
        self,
        company_siret: str,
        bs_data_dfs: Dict[str, pd.DataFrame],
        departements_regions_df: pd.DataFrame,
        regions_geodata: gpd.GeoDataFrame,
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.bs_data_dfs = bs_data_dfs
        self.departements_regions_df = departements_regions_df
        self.regions_geodata = regions_geodata
        self.data_date_interval = data_date_interval

        self.preprocessed_df = None
        self.figure = None

    def _preprocess_data(self) -> None:
        if len(self.bs_data_dfs) == 0:
            return

        concat_df = pd.concat(
            [df[df["received_at"].between(*self.data_date_interval)] for df in self.bs_data_dfs.values()]
        )

        # The postal code is extracted from the address field using a simple regex
        concat_df["cp"] = concat_df["emitter_company_address"].str.extract(r"([0-9]{5})", expand=False)
        concat_df["code_dep"] = concat_df["cp"].apply(get_code_departement)

        if concat_df["code_dep"].isna().all():
            return

        concat_df = pd.merge(
            concat_df,
            self.departements_regions_df,
            left_on="code_dep",
            right_on="DEP",
            how="left",
            validate="many_to_one",
        )

        # Handle quantity refused
        if "quantity_refused" in concat_df.columns:
            concat_df["quantity_received"] = concat_df["quantity_received"] - concat_df["quantity_refused"].fillna(0)

        # The 'Region' label is kept after aggregation
        df_grouped = (
            concat_df[concat_df["recipient_company_siret"] == self.company_siret]
            .groupby("LIBELLE_reg")
            .aggregate({"quantity_received": "sum", "REG": "max"})
        )

        final_df = pd.merge(self.regions_geodata, df_grouped, left_on="code", right_on="REG", how="left")

        final_df.fillna(0, inplace=True)

        self.preprocessed_df = final_df

    def _check_data_empty(self) -> bool:
        if (
            (self.preprocessed_df is None)
            or self.preprocessed_df["quantity_received"].isna().all()
            or (len(self.preprocessed_df) == 0)
            or (self.preprocessed_df["quantity_received"] == 0).all()
        ):
            return True

        return False

    def _create_figure(self) -> None:
        gdf = self.preprocessed_df
        geojson = json.loads(gdf.to_json())

        # The figure is built in two part.
        # The first trace holds the 'region' geometries.
        # This trace doesn't hold preprocessed data.
        trace = go.Choropleth(
            geojson=geojson,
            z=[0] * len(gdf["quantity_received"]),
            locations=gdf.index,
            locationmode="geojson-id",
            colorscale=["#F9F8F6", "#F9F8F6"],
            marker_line_color="#979797",
            hoverinfo="skip",
            showscale=False,
        )

        sizeref = 2.0 * max(gdf["quantity_received"]) / (12**2)

        gdf_nonzero = gdf[gdf["quantity_received"] != 0]

        # This second trace will holds the circles that will be drawn on the map.
        # It is build using preprocessed data (size are relative to the quantity received).
        trace_2 = go.Scattergeo(
            geojson=geojson,
            locations=gdf_nonzero.index,
            locationmode="geojson-id",
            lat=gdf_nonzero.geometry.centroid.y,
            lon=gdf_nonzero.geometry.centroid.x,
            marker_sizeref=sizeref,
            marker_size=gdf_nonzero["quantity_received"],
            marker_sizemin=3,
            mode="markers+text",
            hovertext=[
                f"{e.nom} - <b>{format_number_str(e.quantity_received, precision=2)}t</b>"
                for e in gdf_nonzero.itertuples()
            ],
            hoverinfo="text",
            marker_color="#518FFF",
        )

        fig = go.Figure([trace, trace_2])
        fig.update_layout(
            margin={"b": 0, "t": 0, "r": 0, "l": 0},
            showlegend=False,
            legend_bgcolor="rgba(0,0,0,0)",
            xaxis_fixedrange=True,
            yaxis_fixedrange=True,
            dragmode=False,
        )
        fig.update_geos(
            fitbounds="locations",
            visible=False,
            showframe=False,
            projection_type="mercator",
        )

        self.figure = fig

        self._preprocess_data()

    def build(self):
        self._preprocess_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class ICPEDailyItemProcessor:
    """Component with a figure representing the quantity of waste processed by day for a particular ICPE "rubrique".


    Parameters:
    -----------
    icpe_item_daily_data: DataFrame
        DataFrame containing the waste processed data for a given ICPE "rubrique".
    """

    def __init__(
        self,
        icpe_item_daily_data: pd.DataFrame,
    ) -> None:
        self.icpe_item_daily_data = icpe_item_daily_data

        self.preprocessed_df = None
        self.authorized_quantity = None
        self.mean_quantity = None

        self.figure = None

    def _preprocess_data(self) -> None:
        if (self.icpe_item_daily_data is None) or (len(self.icpe_item_daily_data) == 0):
            return

        df = self.icpe_item_daily_data[["day_of_processing", "processed_quantity"]]
        df = df.sort_values("day_of_processing")

        self.mean_quantity = df["processed_quantity"].mean()

        series = df.set_index("day_of_processing")
        if len(series) > 1:
            series = series.squeeze()
        final_df = series.resample("1d").max().fillna(0).reset_index()

        self.preprocessed_df = final_df
        self.authorized_quantity = self.icpe_item_daily_data["authorized_quantity"].max()

    def _check_data_empty(self) -> bool:
        if (self.preprocessed_df is None) or len(self.preprocessed_df) == 0:
            return True

        return False

    def _create_figure(self) -> None:
        df = self.preprocessed_df
        authorized_quantity = self.authorized_quantity
        trace = go.Scatter(
            x=df["day_of_processing"],
            y=df["processed_quantity"],
            hovertemplate="Le %{x|%d-%m-%Y} : <b>%{y:.2f}t</b> traitées<extra></extra>",
            line_width=1.5,
        )

        fig = go.Figure([trace])

        fig.update_layout(
            margin={"t": 40, "l": 35, "r": 70},
            legend_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
            title={
                "text": f"Quantité moyenne traitée par jour : <b>{format_number_str(self.mean_quantity, 2)}</b> t/j",
                "font_size": 14,
            },
        )

        max_y = df["processed_quantity"].max()
        if not pd.isna(authorized_quantity):
            fig.add_hline(
                y=authorized_quantity,
                line_dash="dot",
                line_color="red",
                line_width=3,
            )
            fig.add_annotation(
                xref="x domain",
                yref="y",
                x=1,
                y=authorized_quantity,
                text=f"Quantité maximale <br>autorisée :<b>{format_number_str(authorized_quantity, 2)}</b> t/jour",
                font_color="red",
                xanchor="left",
                showarrow=False,
                textangle=-90,
                font_size=13,
            )
            if authorized_quantity > max_y:
                max_y = authorized_quantity

        fig.update_yaxes(range=[0, max_y * 1.3], gridcolor="#ccc", title="Quantité traitée en tonnes")

        fig.update_xaxes(
            gridcolor="#ccc",
            zeroline=True,
            linewidth=1,
            linecolor="black",
            title="Date du traitement",
        )

        self.figure = fig

    def build(self):
        self._preprocess_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class ICPEAnnualItemProcessor:
    """Component with a figure representing the cummulative quantity of waste processed by day for a particular ICPE "rubrique".


    Parameters:
    -----------
    icpe_item_daily_data: DataFrame
        DataFrame containing the waste processed data for a given ICPE "rubrique".
    """

    def __init__(
        self,
        icpe_item_daily_data: pd.DataFrame | None,
    ) -> None:
        self.icpe_item_daily_data = icpe_item_daily_data

        self.preprocessed_df = None
        self.authorized_quantity = None

        self.figure = None

    def _preprocess_data(self) -> None:
        if (self.icpe_item_daily_data is None) or (len(self.icpe_item_daily_data) == 0):
            return

        df = self.icpe_item_daily_data[["day_of_processing", "processed_quantity"]]
        df = df.sort_values("day_of_processing")

        df = df.set_index("day_of_processing")
        final_df = df.resample("1d").max().fillna(0).reset_index()
        final_df["quantity_cumsum"] = final_df.groupby(final_df["day_of_processing"].dt.year).cumsum()

        self.preprocessed_df = final_df
        self.authorized_quantity = self.icpe_item_daily_data["authorized_quantity"].max()

    def _check_data_empty(self) -> bool:
        if (self.preprocessed_df is None) or len(self.preprocessed_df) == 0:
            return True

        if self.preprocessed_df["processed_quantity"].sum() == 0:
            return True

        return False

    def _create_figure(self) -> None:
        df = self.preprocessed_df
        authorized_quantity = self.authorized_quantity

        traces = []

        for _, temp_df in df.groupby(df["day_of_processing"].dt.year):
            trace = go.Scatter(
                x=temp_df["day_of_processing"],
                y=temp_df["quantity_cumsum"],
                hovertemplate="Le %{x|%d-%m-%Y} : <b>%{y:.2f}t</b> traitées au total sur l'année<extra></extra>",
                line_width=2,
            )

            traces.append(trace)

        fig = go.Figure(traces)

        for trace in traces[1:]:
            year = trace["x"][0].year
            x = min(trace["x"])
            fig.add_vline(
                line_dash="dot",
                line_color="black",
                line_width=2,
                x=x.timestamp()
                * 1000,  # Due to this bug : https://github.com/plotly/plotly.py/issues/3065, we have to convert to epoch here
                annotation_text=f"{year}",
                annotation_position="top right",
            )

        fig.update_layout(
            margin={"t": 30, "l": 35, "r": 70},
            legend_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        max_y = df["quantity_cumsum"].max()
        if not pd.isna(authorized_quantity):
            fig.add_hline(
                y=authorized_quantity,
                line_dash="dot",
                line_color="red",
                line_width=3,
            )
            fig.add_annotation(
                xref="x domain",
                yref="y",
                x=1,
                y=authorized_quantity,
                text=f"Quantité maximale <br>autorisée : <b>{format_number_str(authorized_quantity, 2)}</b> t/an",
                font_color="red",
                xanchor="left",
                showarrow=False,
                textangle=-90,
                font_size=13,
            )

            if authorized_quantity > 0:
                # Target for 2025
                fig.add_hline(
                    y=authorized_quantity / 2,
                    line_dash="dot",
                    line_color="black",
                    line_width=2,
                )
                fig.add_annotation(
                    xref="x domain",
                    yref="y",
                    x=0.7,
                    y=authorized_quantity / 2,
                    text=f"Objectif 50% pour 2025 : <b>{format_number_str(authorized_quantity / 2, 2)}</b> t/an",
                    font_color="black",
                    xanchor="left",
                    yanchor="bottom",
                    showarrow=False,
                    font_size=12,
                )

            if authorized_quantity > max_y:
                max_y = authorized_quantity

        fig.update_yaxes(
            range=[0, max_y * 1.3],
            gridcolor="#ccc",
            title="Quantité traitée en tonnes<br>(somme cummulée annuellement)",
        )

        fig.update_xaxes(
            range=[
                df["day_of_processing"].min(),
                df["day_of_processing"].max() + pd.Timedelta(value="7d"),
            ],
            gridcolor="#ccc",
            zeroline=True,
            linewidth=1,
            linecolor="black",
            title="Date du traitement",
        )

        self.figure = fig

    def build(self):
        self._preprocess_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class BsdaWorkerQuantityProcessor:
    """Component with a Line Figure of quantities linked to the worker company siret.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bsda_data_df: DataFrame
        DataFrame containing BSDA data.
    bsda_transporters_data_df : DataFrame
        DataFrames containing information about the transported BSDA waste.
    data_date_interval: tuple
        Date interval to filter data.
    """

    def __init__(
        self,
        company_siret: str,
        bsda_data_df: pd.DataFrame,
        bsda_transporters_data_df: pd.DataFrame | None,
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.bsda_data = bsda_data_df
        self.bsda_transporters_data_df = bsda_transporters_data_df
        self.data_date_interval = data_date_interval

        self.quantities_signed_by_worker_by_month = None
        self.quantities_transported_by_month = None
        self.quantities_processed_by_month = None

        self.figure = None

    def _preprocess_bs_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it for plotting."""
        bsda_data = self.bsda_data.copy()
        transport_df = self.bsda_transporters_data_df

        if (bsda_data is None) or (transport_df is None):
            return

        # Handling multimodal
        bsda_data.drop(
            columns=["sent_at"],
            errors="ignore",
            inplace=True,
        )  # To avoid column duplication with transport data

        bsda_data = bsda_data.merge(
            transport_df[["bs_id", "sent_at", "transporter_company_siret"]],
            left_on="id",
            right_on="bs_id",
            how="left",
            validate="one_to_many",
        )

        bsda_data = bsda_data.groupby("id", as_index=False).agg(
            {
                "worker_company_siret": "max",
                "quantity_received": "max",
                "waste_details_quantity": "max",
                "sent_at": "min",
                "processed_at": "min",
                "worker_work_signature_date": "min",
            }
        )

        bsda_data = bsda_data[bsda_data["worker_company_siret"] == self.company_siret]

        if len(bsda_data) == 0:
            return

        bsda_data_filtered = bsda_data[bsda_data["worker_work_signature_date"].between(*self.data_date_interval)]
        if len(bsda_data_filtered) > 0:
            self.quantities_signed_by_worker_by_month = bsda_data_filtered.groupby(
                pd.Grouper(key="worker_work_signature_date", freq="1M")
            )["waste_details_quantity"].sum()

        bsda_data_filtered = bsda_data[bsda_data["sent_at"].between(*self.data_date_interval)]
        if len(bsda_data_filtered) > 0:
            self.quantities_transported_by_month = bsda_data_filtered.groupby(pd.Grouper(key="sent_at", freq="1M"))[
                "quantity_received"
            ].sum()

        bsda_data_filtered = bsda_data[bsda_data["processed_at"].between(*self.data_date_interval)]
        if len(bsda_data_filtered) > 0:
            self.quantities_processed_by_month = bsda_data_filtered.groupby(pd.Grouper(key="processed_at", freq="1M"))[
                "quantity_received"
            ].sum()

    def _check_data_empty(self) -> bool:
        if all(
            (e is None) or (len(e) == 0)
            for e in [
                self.quantities_signed_by_worker_by_month,
                self.quantities_transported_by_month,
                self.quantities_processed_by_month,
            ]
        ):
            return True

        return False

    def _create_figure(self) -> None:
        lines = []

        configs = [
            {
                "data": self.quantities_signed_by_worker_by_month,
                "name": "Signé par l'entreprise de travaux",
                "hover_suffix": "tonnes (estimées)",
                "color": "#66673D",
            },
            {
                "data": self.quantities_transported_by_month,
                "name": "Enlevé par le transporteur",
                "hover_suffix": "tonnes enlevées",
                "color": "#E4794A",
            },
            {
                "data": self.quantities_processed_by_month,
                "name": "Traité",
                "hover_suffix": "tonnes traitées",
                "color": "#60E0EB",
            },
        ]

        tick0_min = None
        max_y = None
        max_points = 0
        for config in configs:
            data = config["data"]
            hover_suffix = config["hover_suffix"]
            if data is not None and len(data) > 0:
                lines.append(
                    go.Scatter(
                        x=data.index,
                        y=data,
                        name=config["name"],
                        mode="lines+markers",
                        hovertext=[
                            f"{index.strftime('%B %y').capitalize()} - <b>{format_number_str(e, 2)}</b> {hover_suffix}"
                            for index, e in data.items()
                        ],
                        marker_color=config["color"],
                        line_color=config["color"],
                        hoverinfo="text",
                    )
                )
                min_ = data.index.min()
                if (tick0_min is None) or (min_ < tick0_min):
                    tick0_min = min_

                max_ = data.max()
                if (max_y is None) or (max_ < max_y):
                    max_y = max_

                if len(data) > max_points:
                    max_points = len(data)

        fig = go.Figure(lines)

        tickangle = 0
        y_legend = -0.07
        if max_points >= 15:
            tickangle = -90
            y_legend = -0.15

        dtick = "M2"
        if not max_points or max_points < 3:
            dtick = "M1"

        tickangle = 0
        y_legend = -0.07
        if max_points and max_points >= 15:
            tickangle = -90
            y_legend = -0.12

        fig.update_layout(
            margin={"t": 20, "l": 35, "r": 5},
            legend={"orientation": "h", "y": y_legend, "x": 0},
            legend_font_size=11,
            legend_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        fig.update_xaxes(
            tickangle=tickangle,
            tickformat="%b %y",
            tick0=tick0_min,
            dtick=dtick,
            gridcolor="#ccc",
        )
        fig.update_yaxes(exponentformat="B", tickformat=".2s", gridcolor="#ccc")

        self.figure = fig

    def build(self) -> str:
        self._preprocess_bs_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class TransporterBordereauxGraphProcessor:
    """Component with a Bar Figure showing monthly number of bordereaux as transporter company.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    transporters_data_df: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau transported data.
        Correspond to the new way of managing transporters in Trackdéchets.
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    data_date_interval: tuple
        Date interval to filter data.
    """

    def __init__(
        self,
        company_siret: str,
        transporters_data_df: Dict[str, pd.DataFrame],  # Handling new multi-modal Trackdéchets feature
        bs_data_dfs: Dict[str, pd.DataFrame],
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.transporters_data_df = transporters_data_df
        self.bs_data_dfs = bs_data_dfs
        self.data_date_interval = data_date_interval

        self.transported_bordereaux_stats = {
            BSDD: None,
            BSDD_NON_DANGEROUS: None,
            BSDA: None,
            BSFF: None,
            BSDASRI: None,
            BSVHU: None,
        }

        self.figure = None

    def _preprocess_bs_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it for plotting."""
        transporter_data_dfs = self.transporters_data_df
        bs_data_dfs = self.bs_data_dfs

        for bs_type, df in chain(transporter_data_dfs.items(), bs_data_dfs.items()):
            df = df[
                df["sent_at"].between(*self.data_date_interval)
                & (df["transporter_company_siret"] == self.company_siret)
            ]

            if len(df) > 0:
                id_col = "bs_id" if bs_type in [BSDD, BSDD_NON_DANGEROUS, BSDA, BSFF] else "id"
                df_by_month = df.groupby(pd.Grouper(key="sent_at", freq="1M"))[id_col].nunique()
                self.transported_bordereaux_stats[bs_type] = df_by_month

    def _check_data_empty(self) -> bool:
        if all((e is None) or (len(e) == 0) for e in self.transported_bordereaux_stats.values()):
            return True

        return False

    def _create_figure(self) -> None:
        bars = []

        configs = [
            {
                "data": self.transported_bordereaux_stats[BSDD],
                "name": "BSDD",
                "hover_suffix": "BSDD transportés",
            },
            {
                "data": self.transported_bordereaux_stats[BSDD_NON_DANGEROUS],
                "name": "BSDD Non Dangereux",
                "hover_suffix": "BSDD Non Dangereux transportés",
            },
            {
                "data": self.transported_bordereaux_stats[BSDA],
                "name": "BSDA",
                "hover_suffix": "BSDA transportés",
            },
            {
                "data": self.transported_bordereaux_stats[BSFF],
                "name": "BSFF",
                "hover_suffix": "BSFF transportés",
            },
            {
                "data": self.transported_bordereaux_stats[BSDASRI],
                "name": "BSDASRI",
                "hover_suffix": "BSDASRI transportés",
            },
            {
                "data": self.transported_bordereaux_stats[BSVHU],
                "name": "BSVHU",
                "hover_suffix": "BSVHU transportés",
            },
        ]

        tick0_min = None
        max_y = None
        max_points = 0
        for config in configs:
            data = config["data"]
            hover_suffix = config["hover_suffix"]
            if data is not None and len(data) > 0:
                bars.append(
                    go.Bar(
                        x=data.index,
                        y=data,
                        text=data,
                        texttemplate="%{text:.0s}",
                        textposition="auto",
                        name=config["name"],
                        hovertext=[
                            f"{index.strftime('%B %y').capitalize()} - <b>{format_number_str(e, 2)}</b> {hover_suffix}"
                            for index, e in data.items()
                        ],
                        hoverinfo="text",
                    )
                )
                min_ = data.index.min()
                if (tick0_min is None) or (min_ < tick0_min):
                    tick0_min = min_

                max_ = data.max()
                if (max_y is None) or (max_ < max_y):
                    max_y = max_

                if len(data) > max_points:
                    max_points = len(data)

        fig = go.Figure(bars)

        tickangle = 0
        y_legend = -0.07
        if max_points >= 15:
            tickangle = -90
            y_legend = -0.15

        dtick = "M2"
        if not max_points or max_points < 3:
            dtick = "M1"

        tickangle = 0
        y_legend = -0.07
        if max_points and max_points >= 15:
            tickangle = -90
            y_legend = -0.12

        fig.update_layout(
            margin={"t": 20, "l": 35, "r": 5},
            legend={"orientation": "h", "y": y_legend, "x": 0},
            legend_font_size=11,
            legend_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="stack",
            margin_pad=5,
        )

        fig.update_xaxes(
            tickangle=tickangle,
            tickformat="%b %y",
            tick0=tick0_min,
            dtick=dtick,
            gridcolor="#ccc",
        )
        fig.update_yaxes(exponentformat="B", tickformat=".2s", gridcolor="#ccc")

        self.figure = fig

    def build(self):
        self._preprocess_bs_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class TransportedQuantitiesGraphProcessor:
    """Component with a Line Figure showing monthly quantity fo waste transported company.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    transporters_data_df: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau transported data.
        Correspond to the new way of managing transporters in Trackdéchets.
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    data_date_interval: tuple
        Date interval to filter data.
    packagings_data_df : pd.DataFrame | None
        Optional parameter that represents a DataFrame containing data about BSFF packagings.
    """

    def __init__(
        self,
        company_siret: str,
        transporters_data_df: Dict[str, pd.DataFrame],  # Handling new multi-modal Trackdéchets feature
        bs_data_dfs: Dict[str, pd.DataFrame],
        data_date_interval: tuple[datetime, datetime],
        packagings_data_df: pd.DataFrame | None = None,
    ) -> None:
        self.company_siret = company_siret
        self.transporters_data_df = transporters_data_df
        self.bs_data_dfs = bs_data_dfs
        self.data_date_interval = data_date_interval
        self.packagings_data_df = packagings_data_df

        self.transported_quantities_stats = {
            BSDD: None,
            BSDD_NON_DANGEROUS: None,
            BSDA: None,
            BSFF: None,
            BSDASRI: None,
            BSVHU: None,
        }

        self.figure = None

    def _preprocess_bs_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it for plotting."""
        transporter_data_dfs = self.transporters_data_df
        bs_data_dfs = self.bs_data_dfs

        for bs_type, df in chain(transporter_data_dfs.items(), bs_data_dfs.items()):
            df = df[
                df["sent_at"].between(*self.data_date_interval)
                & (df["transporter_company_siret"] == self.company_siret)
            ]

            if len(df) > 0:
                df_by_month = df.groupby(pd.Grouper(key="sent_at", freq="1M"))["quantity_received"].sum()
                self.transported_quantities_stats[bs_type] = df_by_month

    def _check_data_empty(self) -> bool:
        if all((e is None) or (len(e) == 0) for e in self.transported_quantities_stats.values()):
            return True

        return False

    def _create_figure(self) -> None:
        bars = []

        configs = [
            {
                "data": self.transported_quantities_stats[BSDD],
                "name": "BSDD",
                "hover_text": "{} - <b>{}</b> tonnes de déchets dangereux transportés",
            },
            {
                "data": self.transported_quantities_stats[BSDD_NON_DANGEROUS],
                "name": "BSDD Non Dangereux",
                "hover_text": "{} - <b>{}</b> tonnes de déchets non dangereux transportés",
            },
            {
                "data": self.transported_quantities_stats[BSDA],
                "name": "BSDA",
                "hover_text": "{} - <b>{}</b> tonnes de déchets amiante transportés",
            },
            {
                "data": self.transported_quantities_stats[BSFF],
                "name": "BSFF",
                "hover_text": "{} - <b>{}</b> tonnes de déchets de fluides frigorigènes transportés",
            },
            {
                "data": self.transported_quantities_stats[BSDASRI],
                "name": "BSDASRI",
                "hover_text": "{} - <b>{}</b> tonnes de DASRI transportés",
            },
            {
                "data": self.transported_quantities_stats[BSVHU],
                "name": "BSVHU",
                "hover_text": "{} - <b>{}</b> tonnes de véhicules hors d'usage transportés",
            },
        ]

        tick0_min = None
        max_y = None
        max_points = 0
        for config in configs:
            data = config["data"]
            hover_text = config["hover_text"]
            if data is not None and len(data) > 0:
                bars.append(
                    go.Scatter(
                        x=data.index,
                        y=data,
                        name=config["name"],
                        mode="lines+markers",
                        hovertext=[
                            hover_text.format(
                                index.strftime("%B %y").capitalize(),
                                format_number_str(e),
                            )
                            for index, e in data.items()
                        ],
                        hoverinfo="text",
                        stackgroup="one",
                    )
                )
                min_ = data.index.min()
                if (tick0_min is None) or (min_ < tick0_min):
                    tick0_min = min_

                max_ = data.max()
                if (max_y is None) or (max_ < max_y):
                    max_y = max_

                if len(data) > max_points:
                    max_points = len(data)

        fig = go.Figure(bars)

        tickangle = 0
        y_legend = -0.07
        if max_points >= 15:
            tickangle = -90
            y_legend = -0.15

        dtick = "M2"
        if not max_points or max_points < 3:
            dtick = "M1"

        tickangle = 0
        y_legend = -0.07
        if max_points and max_points >= 15:
            tickangle = -90
            y_legend = -0.12

        fig.update_layout(
            margin={"t": 20, "l": 35, "r": 5},
            legend={"orientation": "h", "y": y_legend, "x": 0},
            legend_font_size=11,
            legend_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
            margin_pad=5,
        )

        fig.update_xaxes(
            tickangle=tickangle,
            tickformat="%b %y",
            tick0=tick0_min,
            dtick=dtick,
            gridcolor="#ccc",
        )
        fig.update_yaxes(exponentformat="B", tickformat=".2s", gridcolor="#ccc", ticksuffix="t")

        self.figure = fig

    def build(self):
        self._preprocess_bs_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class RNDTSQuantitiesGraphProcessor:
    """Component with a Line Figure showing incoming and outgoing quantities of non dangerous waste.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    rndts_incoming_data: DataFrame
        DataFrame containing data for incoming non dangerous waste (from RNDTS).
    rndts_outgoing_data: DataFrame
        DataFrame containing data for outgoing non dangerous waste (from RNDTS).
    data_date_interval: tuple
        Date interval to filter data.
    """

    def __init__(
        self,
        company_siret: str,
        rndts_incoming_data: pd.DataFrame | None,
        rndts_outgoing_data: pd.DataFrame | None,
        data_date_interval: tuple[datetime, datetime],
    ):
        self.company_siret = company_siret
        self.rndts_incoming_data = rndts_incoming_data
        self.rndts_outgoing_data = rndts_outgoing_data
        self.data_date_interval = data_date_interval

        self.incoming_weight_by_month_serie = pd.Series()
        self.outgoing_weight_by_month_serie = pd.Series()

        self.incoming_volume_by_month_serie = pd.Series()
        self.outgoing_volume_by_month_serie = pd.Series()

        self.figure = None

    def _preprocess_data(self) -> None:
        # We need to account for quantities in m³ and t

        incoming_data = self.rndts_incoming_data
        if (incoming_data is not None) and (len(incoming_data) > 0):
            incoming_data = incoming_data[
                (incoming_data["date_reception"].between(*self.data_date_interval))
                & (incoming_data["etablissement_numero_identification"] == self.company_siret)
            ]

            if len(incoming_data) > 0:
                self.incoming_weight_by_month_serie = (
                    incoming_data[incoming_data["unite"] == "T"]
                    .groupby(pd.Grouper(key="date_reception", freq="1M"))["quantite"]
                    .sum()
                    .replace(0, np.nan)
                )
                self.incoming_volume_by_month_serie = (
                    incoming_data[incoming_data["unite"] == "M3"]
                    .groupby(pd.Grouper(key="date_reception", freq="1M"))["quantite"]
                    .sum()
                    .replace(0, np.nan)
                )

        outgoing_data = self.rndts_outgoing_data
        if (outgoing_data is not None) and (len(outgoing_data) > 0):
            colname = "producteur_numero_identification"
            if "producteur_numero_identification" not in outgoing_data.columns:
                colname = "etablissement_numero_identification"  # SSD case

            outgoing_data = outgoing_data[
                outgoing_data["date_expedition"].between(*self.data_date_interval)
                & (outgoing_data[colname] == self.company_siret)
            ]

            if len(outgoing_data) > 0:
                self.outgoing_weight_by_month_serie = (
                    outgoing_data[outgoing_data["unite"] == "T"]
                    .groupby(pd.Grouper(key="date_expedition", freq="1M"))["quantite"]
                    .sum()
                    .replace(0, np.nan)
                )
                self.outgoing_volume_by_month_serie = (
                    outgoing_data[outgoing_data["unite"] == "M3"]
                    .groupby(pd.Grouper(key="date_expedition", freq="1M"))["quantite"]
                    .sum()
                    .replace(0, np.nan)
                )

    def _check_data_empty(self) -> bool:
        series = [
            self.incoming_weight_by_month_serie,
            self.outgoing_weight_by_month_serie,
            self.incoming_volume_by_month_serie,
            self.outgoing_volume_by_month_serie,
        ]

        # If DataFrames are empty then output is empty
        if all(len(s) == 0 for s in series):
            return True

        # If preprocessed series are full of NA then output is empty
        if all(s.isna().all() for s in series):
            return True

        # If preprocessed series are full of 0's then output is empty
        if all((s == 0).all() for s in series):
            return True

        return False

    def _create_figure(self) -> None:
        fig = go.Figure()

        lines = []  # Will store the lines graph objects

        # We store the minimum date of each series to be able to configure
        # the tick 0 of the figure
        mins_x = []

        # This is used to configure the dticks in case of low number of data points.
        numbers_of_data_points = []

        # We create two lines (for incoming and outgoing) for each quantity variable chosen
        for variable_name, incoming_data_by_month, outgoing_data_by_month in zip(
            ["quantite", "volume"],
            [
                self.incoming_weight_by_month_serie,
                self.incoming_volume_by_month_serie,
            ],
            [self.outgoing_weight_by_month_serie, self.outgoing_volume_by_month_serie],
        ):
            incoming_line_name = "Quantité entrante (t)"
            incoming_hover_text = "{} - <b>{}</b> tonnes entrantes"
            outgoing_line_name = "Quantité sortante (t)"
            outgoing_hover_text = "{} - <b>{}</b> tonnes sortantes"
            marker_line_style = "solid"
            marker_symbol = "circle"
            marker_size = 6

            # To handle the case of volume
            if variable_name == "volume":
                incoming_line_name = "Volume entrant (m³)"
                incoming_hover_text = "{} - <b>{}</b> m³ entrants"
                outgoing_line_name = "Volume sortant (m³)"
                outgoing_hover_text = "{} - <b>{}</b> m³ sortants"
                marker_line_style = "dash"
                marker_symbol = "triangle-up"
                marker_size = 10

            if len(incoming_data_by_month) > 0:
                incoming_line = go.Scatter(
                    x=incoming_data_by_month.index,
                    y=incoming_data_by_month,
                    name=incoming_line_name,
                    mode="lines+markers",
                    hovertext=[
                        incoming_hover_text.format(index.strftime("%B %y").capitalize(), format_number_str(e))
                        for index, e in incoming_data_by_month.items()
                    ],
                    hoverinfo="text",
                    marker_color="#E1000F",
                    marker_symbol=marker_symbol,
                    marker_size=marker_size,
                    line_dash=marker_line_style,
                )
                mins_x.append(incoming_data_by_month.index.min())
                numbers_of_data_points.append(len(incoming_data_by_month))
                lines.append(incoming_line)

            if len(outgoing_data_by_month) > 0:
                outgoing_line = go.Scatter(
                    x=outgoing_data_by_month.index,
                    y=outgoing_data_by_month,
                    name=outgoing_line_name,
                    mode="lines+markers",
                    hovertext=[
                        outgoing_hover_text.format(index.strftime("%B %y").capitalize(), format_number_str(e))
                        for index, e in outgoing_data_by_month.items()
                    ],
                    hoverinfo="text",
                    marker_color="#6A6AF4",
                    marker_symbol=marker_symbol,
                    marker_size=marker_size,
                    line_dash=marker_line_style,
                )
                mins_x.append(outgoing_data_by_month.index.min())
                numbers_of_data_points.append(len(outgoing_data_by_month))
                lines.append(outgoing_line)

        fig.add_traces(lines)

        dtick = "M2"
        if not numbers_of_data_points or max(numbers_of_data_points) < 3:
            dtick = "M1"

        tickangle = 0
        y_legend = -0.07
        if numbers_of_data_points and max(numbers_of_data_points) >= 15:
            tickangle = -90
            y_legend = -0.12

        fig.update_layout(
            margin={"t": 20, "l": 35, "r": 5},
            legend={"orientation": "h", "y": y_legend, "x": 0},
            legend_font_size=11,
            legend_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        fig.update_xaxes(
            tickangle=tickangle,
            tickformat="%b %y",
            tick0=min(mins_x) if mins_x else None,
            dtick=dtick,
            gridcolor="#ccc",
        )
        fig.update_yaxes(exponentformat="B", tickformat=".2s", gridcolor="#ccc")

        self.figure = fig

    def build(self):
        self._preprocess_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class RNDTSStatementsGraphProcessor:
    """Component with a Bar Figure of incoming and outgoing RNDTS statements.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    rndts_incoming_data: DataFrame
        DataFrame containing data for incoming non dangerous waste (from RNDTS).
    rndts_outgoing_data: DataFrame
        DataFrame containing data for outgoing non dangerous waste (from RNDTS).
    statement_type: str
        Type of statement used as input, either non dangerous waste statements, excavated lands statements or ssd.
    data_date_interval: tuple
        Date interval to filter data.
    """

    def __init__(
        self,
        company_siret: str,
        rndts_incoming_data: pd.DataFrame | None,
        rndts_outgoing_data: pd.DataFrame | None,
        statement_type: Literal["non_dangerous_waste"] | Literal["excavated_land"] | Literal["ssd"],
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.rndts_incoming_data = rndts_incoming_data
        self.rndts_outgoing_data = rndts_outgoing_data
        self.statement_type = statement_type
        self.data_date_interval = data_date_interval

        self.statements_emitted_by_month_serie = None
        self.statements_received_by_month_serie = None

        self.figure = None

    def _preprocess_bs_data(self) -> None:
        """Preprocess raw RNDTS data to prepare it for plotting."""

        incoming_data = self.rndts_incoming_data
        if (incoming_data is not None) and (len(incoming_data) > 0):
            incoming_data = incoming_data[
                incoming_data["date_reception"].between(*self.data_date_interval)
                & (incoming_data["etablissement_numero_identification"] == self.company_siret)
            ].dropna(subset=["date_reception"])

            if len(incoming_data) > 0:
                self.statements_received_by_month_serie = incoming_data.groupby(
                    pd.Grouper(key="date_reception", freq="1M")
                ).id.count()

        outgoing_data = self.rndts_outgoing_data
        if (outgoing_data is not None) and (len(outgoing_data) > 0):
            colname = "producteur_numero_identification"
            if "producteur_numero_identification" not in outgoing_data.columns:
                colname = "etablissement_numero_identification"  # SSD case
            outgoing_data = outgoing_data[
                outgoing_data["date_expedition"].between(*self.data_date_interval)
                & (outgoing_data[colname] == self.company_siret)
            ].dropna(subset=["date_expedition"])

            if len(outgoing_data) > 0:
                self.statements_emitted_by_month_serie = outgoing_data.groupby(
                    pd.Grouper(key="date_expedition", freq="1M")
                ).id.count()

    def _check_data_empty(self) -> bool:
        match [self.statements_emitted_by_month_serie, self.statements_received_by_month_serie]:
            case [None, None]:
                return True
            case [df, None] | [None, df]:
                if len(df) == 0:
                    return True
            case [df1, df2]:
                if len(df1) == len(df2) == 0:
                    return True

        return False

    def _create_figure(self) -> None:
        statements_emitted_by_month = self.statements_emitted_by_month_serie
        statements_received_by_month = self.statements_received_by_month_serie

        text_size = 12

        bars = []
        ticks0 = []
        nums_points = []
        # Used to store the maximum value of each line
        # to be able to configure the height of the plotting area of the figure.
        max_y = 0

        match self.statement_type:
            case "non_dangerous_waste":
                name = "DND"
                hover_suffix = "déchets non dangereux"
            case "excavated_land":
                name = "TEXS"
                hover_suffix = "TEXS"
            case "ssd":
                name = "SSD"
                hover_suffix = "sorties de statut de déchet"
            case _:
                hover_suffix = ""

        if (statements_emitted_by_month is not None) and (len(statements_emitted_by_month) > 0):
            statements_emitted_bars = go.Bar(
                x=statements_emitted_by_month.index,
                y=statements_emitted_by_month,
                name=f"{name} - sortant",
                hovertext=[
                    "{} - <b>{}</b> déclaration(s) sortante(s) de {}".format(
                        index.strftime("%B %y").capitalize(), e, hover_suffix
                    )
                    for index, e in statements_emitted_by_month.items()
                ],
                hoverinfo="text",
                textfont_size=text_size,
                textposition="outside",
                constraintext="none",
                marker_color="#6A6AF4",
            )
            ticks0.append(statements_emitted_by_month.index.min())
            max_y = max(max_y, statements_emitted_by_month.max())
            nums_points.append(len(statements_emitted_by_month))
            bars.append(statements_emitted_bars)

        if (statements_received_by_month is not None) and (len(statements_received_by_month) > 0):
            statements_received_bars = go.Bar(
                x=statements_received_by_month.index,
                y=statements_received_by_month,
                name=f"{name} - entrant",
                hovertext=[
                    "{} - <b>{}</b> déclaration(s) de {}".format(index.strftime("%B %y").capitalize(), e, hover_suffix)
                    for index, e in statements_received_by_month.items()
                ],
                hoverinfo="text",
                textfont_size=text_size,
                textposition="outside",
                constraintext="none",
                marker_color="#E1000F",
            )
            ticks0.append(statements_received_by_month.index.min())
            max_y = max(max_y, statements_received_by_month.max())
            nums_points.append(len(statements_received_by_month))
            bars.append(statements_received_bars)

        tick0_min = min(ticks0)

        fig = go.Figure(bars)

        max_points = max(nums_points)

        tickangle = 0
        y_legend = -0.07
        if max_points >= 15:
            tickangle = -90
            y_legend = -0.15

        fig.update_layout(
            margin={"t": 20, "l": 35, "r": 5},
            legend={
                "orientation": "h",
                "y": y_legend,
                "x": -0.1,
            },
            legend_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        ticklabelstep = 2
        if max_points <= 3:
            ticklabelstep = 1

        fig.update_xaxes(
            dtick=f"M{ticklabelstep}",
            tickangle=tickangle,
            tickformat="%b %y",
            tick0=tick0_min,
            ticks="outside",
            gridcolor="#ccc",
        )

        # Range of the y axis is increased to increase the height of the plotting are of the figure
        fig.update_yaxes(range=[0, max_y * 1.1], gridcolor="#ccc")

        self.figure = fig

    def build(self):
        self._preprocess_bs_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class IntermediaryBordereauxCountsGraphProcessor:
    """Component with a Bar Figure showing monthly number of bordereaux sent
    as "eco-organisme" or other type of intermediary.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    transporters_data_df: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau transported data.
        Correspond to the new way of managing transporters in Trackdéchets.
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    data_date_interval: tuple
        Date interval to filter data.
    """

    def __init__(
        self,
        company_siret: str,
        bs_data_dfs: Dict[str, pd.DataFrame],
        transporters_data_df: Dict[str, pd.DataFrame],  # Handling new multi-modal Trackdéchets feature
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.bs_data_dfs = bs_data_dfs
        self.transporters_data_df = transporters_data_df
        self.data_date_interval = data_date_interval

        self.bordereaux_stats = {
            BSDD: None,
            BSDD_NON_DANGEROUS: None,
            BSDA: None,
            BSDASRI: None,
        }

        self.figure = None

    def _preprocess_bs_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it for plotting."""
        bs_data_dfs = self.bs_data_dfs

        for bs_type, df in bs_data_dfs.items():
            df = df.copy()
            if bs_type in [BSDD, BSDD_NON_DANGEROUS, BSDA]:
                transport_df = self.transporters_data_df.get(bs_type)

                if (transport_df is None) or len(transport_df) == 0:
                    continue

                df.drop(
                    columns=["sent_at"], errors="ignore", inplace=True
                )  # To avoid column duplication with transport data

                df = df.merge(
                    transport_df[["bs_id", "sent_at"]],
                    left_on="id",
                    right_on="bs_id",
                    how="left",
                    validate="one_to_many",
                )

            df = df[
                df["sent_at"].between(*self.data_date_interval) & (df["eco_organisme_siret"] == self.company_siret)
            ]

            if len(df) > 0:
                df_by_month = df.groupby(pd.Grouper(key="sent_at", freq="1M"))["id"].nunique()
                self.bordereaux_stats[bs_type] = df_by_month

    def _check_data_empty(self) -> bool:
        if all((e is None) or (len(e) == 0) for e in self.bordereaux_stats.values()):
            return True

        return False

    def _create_figure(self) -> None:
        bars = []

        configs = [
            {
                "data": self.bordereaux_stats[BSDD],
                "name": "BSDD",
                "hover_suffix": "BSDD traités",
            },
            {
                "data": self.bordereaux_stats[BSDD_NON_DANGEROUS],
                "name": "BSDD Non Dangereux",
                "hover_suffix": "BSDD Non Dangereux traités",
            },
            {
                "data": self.bordereaux_stats[BSDA],
                "name": "BSDA",
                "hover_suffix": "BSDA traités",
            },
            {
                "data": self.bordereaux_stats[BSDASRI],
                "name": "BSDASRI",
                "hover_suffix": "BSDASRI traités",
            },
        ]

        tick0_min = None
        max_y = None
        max_points = 0
        for config in configs:
            data = config["data"]
            hover_suffix = config["hover_suffix"]
            if data is not None and len(data) > 0:
                bars.append(
                    go.Bar(
                        x=data.index,
                        y=data,
                        text=data,
                        texttemplate="%{text:.0s}",
                        textposition="auto",
                        name=config["name"],
                        hovertext=[
                            f"{index.strftime('%B %y').capitalize()} - <b>{format_number_str(e, 2)}</b> {hover_suffix}"
                            for index, e in data.items()
                        ],
                        hoverinfo="text",
                    )
                )
                min_ = data.index.min()
                if (tick0_min is None) or (min_ < tick0_min):
                    tick0_min = min_

                max_ = data.max()
                if (max_y is None) or (max_ < max_y):
                    max_y = max_

                if len(data) > max_points:
                    max_points = len(data)

        fig = go.Figure(bars)

        tickangle = 0
        y_legend = -0.07
        if max_points >= 15:
            tickangle = -90
            y_legend = -0.15

        dtick = "M2"
        if not max_points or max_points < 3:
            dtick = "M1"

        tickangle = 0
        y_legend = -0.07
        if max_points and max_points >= 15:
            tickangle = -90
            y_legend = -0.12

        fig.update_layout(
            margin={"t": 20, "l": 35, "r": 5},
            legend={"orientation": "h", "y": y_legend, "x": 0},
            legend_font_size=11,
            legend_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="stack",
            margin_pad=5,
        )

        fig.update_xaxes(
            tickangle=tickangle,
            tickformat="%b %y",
            tick0=tick0_min,
            dtick=dtick,
            gridcolor="#ccc",
        )
        fig.update_yaxes(exponentformat="B", tickformat=".2s", gridcolor="#ccc")

        self.figure = fig

    def build(self):
        self._preprocess_bs_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class IntermediaryBordereauxQuantitiesGraphProcessor:
    """Component with a Bar Figure showing monthly number of bordereaux sent
    as "eco-organisme" or other type of intermediary.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    transporters_data_df: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau transported data.
        Correspond to the new way of managing transporters in Trackdéchets.
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    data_date_interval: tuple
        Date interval to filter data.
    """

    def __init__(
        self,
        company_siret: str,
        bs_data_dfs: Dict[str, pd.DataFrame],
        transporters_data_df: Dict[str, pd.DataFrame],  # Handling new multi-modal Trackdéchets feature
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.bs_data_dfs = bs_data_dfs
        self.transporters_data_df = transporters_data_df
        self.data_date_interval = data_date_interval

        self.bordereaux_stats = {
            BSDD: None,
            BSDD_NON_DANGEROUS: None,
            BSDA: None,
            BSDASRI: None,
        }

        self.figure = None

    def _preprocess_bs_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it for plotting."""
        bs_data_dfs = self.bs_data_dfs

        for bs_type, df in bs_data_dfs.items():
            df = df.copy()
            if bs_type in [BSDD, BSDD_NON_DANGEROUS, BSDA]:
                transport_df = self.transporters_data_df.get(bs_type)

                if (transport_df is None) or len(transport_df) == 0:
                    continue

                df.drop(
                    columns=["sent_at"], errors="ignore", inplace=True
                )  # To avoid column duplication with transport data

                df = df.merge(
                    transport_df[["bs_id", "sent_at"]],
                    left_on="id",
                    right_on="bs_id",
                    how="left",
                    validate="one_to_many",
                )

            df = df[
                df["sent_at"].between(*self.data_date_interval) & (df["eco_organisme_siret"] == self.company_siret)
            ]
            df = df.drop_duplicates("id")

            if len(df) > 0:
                if bs_type in [BSDD, BSDD_NON_DANGEROUS, BSDA, BSDASRI]:
                    df["quantity_received"] = df["quantity_received"] - df["quantity_refused"].fillna(0)

                df_by_month = df.groupby(pd.Grouper(key="sent_at", freq="1M"))["quantity_received"].sum()
                self.bordereaux_stats[bs_type] = df_by_month

    def _check_data_empty(self) -> bool:
        if all((e is None) or (len(e) == 0) for e in self.bordereaux_stats.values()):
            return True

        return False

    def _create_figure(self) -> None:
        bars = []

        configs = [
            {
                "data": self.bordereaux_stats[BSDD],
                "name": "BSDD",
                "hover_text": "{} - <b>{}</b> tonnes de déchets dangereux",
            },
            {
                "data": self.bordereaux_stats[BSDD_NON_DANGEROUS],
                "name": "BSDD Non Dangereux",
                "hover_text": "{} - <b>{}</b> tonnes de déchets non dangereux",
            },
            {
                "data": self.bordereaux_stats[BSDA],
                "name": "BSDA",
                "hover_text": "{} - <b>{}</b> tonnes de déchets amiante",
            },
            {
                "data": self.bordereaux_stats[BSDASRI],
                "name": "BSDASRI",
                "hover_text": "{} - <b>{}</b> tonnes de DASRI",
            },
        ]

        tick0_min = None
        max_y = None
        max_points = 0
        for config in configs:
            data = config["data"]
            hover_text = config["hover_text"]
            if data is not None and len(data) > 0:
                bars.append(
                    go.Scatter(
                        x=data.index,
                        y=data,
                        name=config["name"],
                        mode="lines+markers",
                        hovertext=[
                            hover_text.format(
                                index.strftime("%B %y").capitalize(),
                                format_number_str(e),
                            )
                            for index, e in data.items()
                        ],
                        hoverinfo="text",
                        stackgroup="one",
                    )
                )
                min_ = data.index.min()
                if (tick0_min is None) or (min_ < tick0_min):
                    tick0_min = min_

                max_ = data.max()
                if (max_y is None) or (max_ < max_y):
                    max_y = max_

                if len(data) > max_points:
                    max_points = len(data)

        fig = go.Figure(bars)

        tickangle = 0
        y_legend = -0.07
        if max_points >= 15:
            tickangle = -90
            y_legend = -0.15

        dtick = "M2"
        if not max_points or max_points < 3:
            dtick = "M1"

        tickangle = 0
        y_legend = -0.07
        if max_points and max_points >= 15:
            tickangle = -90
            y_legend = -0.12

        fig.update_layout(
            margin={"t": 20, "l": 35, "r": 5},
            legend={"orientation": "h", "y": y_legend, "x": 0},
            legend_font_size=11,
            legend_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
            margin_pad=5,
        )

        fig.update_xaxes(
            tickangle=tickangle,
            tickformat="%b %y",
            tick0=tick0_min,
            dtick=dtick,
            gridcolor="#ccc",
        )
        fig.update_yaxes(exponentformat="B", tickformat=".2s", gridcolor="#ccc", ticksuffix="t")

        self.figure = fig

    def build(self):
        self._preprocess_bs_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class RNDTSTransporterStatementsStatsGraphProcessor:
    """Component with a Bar Figure showing monthly number of RNDTS statements as transporter company.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    rndts_data: dict
        Dict with key being the 'RNDTS' data type and values the DataFrame containing the statements data.
    data_date_interval: tuple
        Date interval to filter data.
    """

    def __init__(
        self,
        company_siret: str,
        rndts_data: Dict[str, pd.DataFrame],
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.rndts_data = rndts_data
        self.data_date_interval = data_date_interval

        self.transported_statements_stats = {
            "ndw_incoming": None,
            "ndw_outgoing": None,
            "excavated_land_incoming": None,
            "excavated_land_outgoing": None,
        }

        self.figure = None

    def _preprocess_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it for plotting."""
        rndts_data = self.rndts_data

        for key, date_col in [
            ("ndw_incoming", "date_reception"),
            ("ndw_outgoing", "date_expedition"),
            ("excavated_land_incoming", "date_reception"),
            ("excavated_land_outgoing", "date_expedition"),
        ]:
            df = rndts_data[key]

            if df is None:
                continue

            df = df[
                df[date_col].between(*self.data_date_interval)
                & (df["numeros_indentification_transporteurs"].apply(lambda x: self.company_siret in x))
            ]

            if len(df) > 0:
                df_by_month = df.groupby(pd.Grouper(key=date_col, freq="1M"))["id"].nunique()
                self.transported_statements_stats[key] = df_by_month

    def _check_data_empty(self) -> bool:
        if all((e is None) or (len(e) == 0) for e in self.transported_statements_stats.values()):
            return True

        return False

    def _create_figure(self) -> None:
        bars = []

        configs = [
            {
                "data": self.transported_statements_stats["ndw_incoming"],
                "name": "DND transportés (entrant)",
                "hover_suffix": "déclaration(s) de DND transportés (registre entrant)",
            },
            {
                "data": self.transported_statements_stats["ndw_outgoing"],
                "name": "DND transportés (sortant)",
                "hover_suffix": "déclaration(s) de DND transportés (registre sortant)",
            },
            {
                "data": self.transported_statements_stats["excavated_land_incoming"],
                "name": "TEXS transportés (entrant)",
                "hover_suffix": "déclaration(s) de TEXS transportés (registre entrant)",
            },
            {
                "data": self.transported_statements_stats["excavated_land_outgoing"],
                "name": "TEXS transportés (sortant)",
                "hover_suffix": "déclaration(s) de TEXS transportés (registre sortant)",
            },
        ]

        tick0_min = None
        max_y = None
        max_points = 0
        for config in configs:
            data = config["data"]
            hover_suffix = config["hover_suffix"]
            if data is not None and len(data) > 0:
                bars.append(
                    go.Bar(
                        x=data.index,
                        y=data,
                        text=data,
                        texttemplate="%{text:.0s}",
                        textposition="auto",
                        name=config["name"],
                        hovertext=[
                            f"{index.strftime('%B %y').capitalize()} - <b>{format_number_str(e, 2)}</b> {hover_suffix}"
                            for index, e in data.items()
                        ],
                        hoverinfo="text",
                    )
                )
                min_ = data.index.min()
                if (tick0_min is None) or (min_ < tick0_min):
                    tick0_min = min_

                max_ = data.max()
                if (max_y is None) or (max_ < max_y):
                    max_y = max_

                if len(data) > max_points:
                    max_points = len(data)

        fig = go.Figure(bars)

        tickangle = 0
        y_legend = -0.07
        if max_points >= 15:
            tickangle = -90
            y_legend = -0.15

        dtick = "M2"
        if not max_points or max_points < 3:
            dtick = "M1"

        tickangle = 0
        y_legend = -0.07
        if max_points and max_points >= 15:
            tickangle = -90
            y_legend = -0.12

        fig.update_layout(
            margin={"t": 20, "l": 35, "r": 5},
            legend={"orientation": "h", "y": y_legend, "x": 0},
            legend_font_size=11,
            legend_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="stack",
            margin_pad=5,
        )

        fig.update_xaxes(
            tickangle=tickangle,
            tickformat="%b %y",
            tick0=tick0_min,
            dtick=dtick,
            gridcolor="#ccc",
        )
        fig.update_yaxes(exponentformat="B", tickformat=".2s", gridcolor="#ccc")

        self.figure = fig

    def build(self):
        self._preprocess_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure


class RNDTSTransporterQuantitiesGraphProcessor:
    """Component with a Bar Figure showing monthly number of RNDTS waste quantitied transported.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    rndts_data: dict
        Dict with key being the 'RNDTS' data type and values the DataFrame containing the statements data.
    data_date_interval: tuple
        Date interval to filter data.
    """

    def __init__(
        self,
        company_siret: str,
        rndts_data: Dict[str, pd.DataFrame],
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.rndts_data = rndts_data
        self.data_date_interval = data_date_interval

        self.transported_quantities_stats = {
            "ndw_incoming": {"T": None, "M3": None},
            "ndw_outgoing": {"T": None, "M3": None},
            "excavated_land_incoming": {"T": None, "M3": None},
            "excavated_land_outgoing": {"T": None, "M3": None},
        }

        self.figure = None

    def _preprocess_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it for plotting."""
        rndts_data = self.rndts_data

        for key, date_col in [
            ("ndw_incoming", "date_reception"),
            ("ndw_outgoing", "date_expedition"),
            ("excavated_land_incoming", "date_reception"),
            ("excavated_land_outgoing", "date_expedition"),
        ]:
            df = rndts_data[key]

            if df is None:
                continue

            for unit in ["T", "M3"]:  # Handle multiple units
                df = df[
                    (df["unite"] == unit)
                    & df[date_col].between(*self.data_date_interval)
                    & (df["numeros_indentification_transporteurs"].apply(lambda x: self.company_siret in x))
                ]

                if len(df) > 0:
                    df_by_month = df.groupby(pd.Grouper(key=date_col, freq="1M"))["quantite"].sum()
                    self.transported_quantities_stats[key][unit] = df_by_month

    def _check_data_empty(self) -> bool:
        if all(
            (ee is None) or (len(ee) == 0) for e in self.transported_quantities_stats.values() for ee in e.values()
        ):
            return True

        return False

    def _create_figure(self) -> None:
        bars = []

        configs = [
            {
                "data": self.transported_quantities_stats["ndw_incoming"],
                "name": "DND transportés (entrant)",
                "hover_suffix": "de DND transportés (registre entrant)",
            },
            {
                "data": self.transported_quantities_stats["ndw_outgoing"],
                "name": "DND transportés (sortant)",
                "hover_suffix": "de DND transportés (registre sortant)",
            },
            {
                "data": self.transported_quantities_stats["excavated_land_incoming"],
                "name": "TEXS transportés (entrant)",
                "hover_suffix": "de TEXS transportés (registre entrant)",
            },
            {
                "data": self.transported_quantities_stats["excavated_land_outgoing"],
                "name": "TEXS transportés (sortant)",
                "hover_suffix": "de TEXS transportés (registre sortant)",
            },
        ]

        tick0_min = None
        max_y = None
        max_points = 0
        for config in configs:
            data = config["data"]
            hover_suffix = config["hover_suffix"]
            if data != {}:
                for unit, data_df in data.items():
                    if (data_df is None) or len(data_df) == 0:
                        continue

                    unit_str = "t"
                    unit_name_str = "masse"
                    marker_line_style = "solid"
                    marker_symbol = "circle"
                    marker_size = 6
                    if unit == "M3":
                        unit_str = "m³"
                        unit_name_str = "volume"
                        marker_line_style = "dash"
                        marker_symbol = "triangle-up"
                        marker_size = 10

                    bars.append(
                        go.Scatter(
                            x=data_df.index,
                            y=data_df,
                            name=f"{unit_name_str.capitalize()} de {config['name']}",
                            mode="lines+markers",
                            hovertext=[
                                f"{index.strftime('%B %y').capitalize()} - <b>{format_number_str(e, 2)}<b>{unit_str} {hover_suffix}"
                                for index, e in data_df.items()
                            ],
                            hoverinfo="text",
                            stackgroup="one",
                            line_dash=marker_line_style,
                            marker_symbol=marker_symbol,
                            marker_size=marker_size,
                        )
                    )
                    min_ = data_df.index.min()
                    if (tick0_min is None) or (min_ < tick0_min):
                        tick0_min = min_

                    max_ = data_df.max()
                    if (max_y is None) or (max_ < max_y):
                        max_y = max_

                    if len(data_df) > max_points:
                        max_points = len(data_df)

        fig = go.Figure(bars)

        tickangle = 0
        y_legend = -0.07
        if max_points >= 15:
            tickangle = -90
            y_legend = -0.15

        dtick = "M2"
        if not max_points or max_points < 3:
            dtick = "M1"

        tickangle = 0
        y_legend = -0.07
        if max_points and max_points >= 15:
            tickangle = -90
            y_legend = -0.12

        fig.update_layout(
            margin={"t": 20, "l": 35, "r": 5},
            legend={"orientation": "h", "y": y_legend, "x": 0},
            legend_font_size=11,
            legend_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            paper_bgcolor="#fff",
            plot_bgcolor="rgba(0,0,0,0)",
            margin_pad=5,
        )

        fig.update_xaxes(
            tickangle=tickangle,
            tickformat="%b %y",
            tick0=tick0_min,
            dtick=dtick,
            gridcolor="#ccc",
        )
        fig.update_yaxes(exponentformat="B", tickformat=".2s", gridcolor="#ccc", ticksuffix="t")

        self.figure = fig

    def build(self):
        self._preprocess_data()

        figure = {}
        if not self._check_data_empty():
            self._create_figure()
            figure = self.figure.to_json()

        return figure
