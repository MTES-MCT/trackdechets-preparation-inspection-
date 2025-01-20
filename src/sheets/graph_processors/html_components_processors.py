import json
import numbers
from datetime import datetime
from itertools import chain
from typing import Dict

import numpy as np
import pandas as pd

from sheets.utils import format_number_str

from ..constants import BS_TYPES_WITH_MULTIMODAL_TRANSPORT, BSDA, BSDASRI, BSDD, BSDD_NON_DANGEROUS, BSFF, BSVHU

# classes returning a context to be rendered in a non-plotly template


class BsdStatsProcessor:
    """Component that displays aggregated data about 'bordereaux' and estimations of the onsite waste stock.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bs_data: DataFrame
        DataFrame containing data for a given 'bordereau' type.
    quantity_variables_names: list of str
        The names of the variables to use to compute quantity statistics.
        For example : ["quantity_received","volume"] to compute statistics for both variables.
    bs_revised_data: DataFrame
        DataFrame containing list of revised 'bordereaux' for a given 'bordereau' type.
    packagings_data:
        For BSFF data, packagings dataset to be able to compute stats at packaging level.
    """

    def __init__(
        self,
        company_siret: str,
        bs_type: str,
        bs_data: pd.DataFrame,
        data_date_interval: tuple[datetime, datetime],
        quantity_variables_names: list[str] = ["quantity_received"],
        bs_revised_data: pd.DataFrame | None = None,
        packagings_data: pd.DataFrame | None = None,
    ) -> None:
        self.company_siret = company_siret
        self.bs_type = bs_type
        self.bs_data = bs_data
        self.data_date_interval = data_date_interval
        self.quantity_variables_names = self._validate_quantity_variables_names(
            quantity_variables_names, packagings_data
        )
        self.bs_revised_data = bs_revised_data
        self.packagings_data = packagings_data

        # Initialization of dicts that will hold the different computed statistics
        keys = [
            "total",
            "archived",
            "processed_in_more_than_one_month_count",
            "processed_in_more_than_one_month_avg_processing_time",
        ]
        if self.packagings_data is not None:
            keys.extend(
                [
                    "total_packagings",
                    "processed_in_more_than_one_month_packagings_count",
                    "processed_in_more_than_one_month_packagings_avg_processing_time",
                ]
            )

        self.emitted_bs_stats = {key: None for key in keys}
        self.received_bs_stats = {key: None for key in keys}

        self.pending_revisions_count = 0
        self.revised_bs_count = 0

        # Quantities stats is two level deep as it will store the statistics for each
        # chosen quantity variables
        self.quantities_stats = {
            key: {
                "total_quantity_incoming": None,
                "total_quantity_outgoing": None,
                "bar_size_incoming": None,
                "bar_size_outgoing": None,
            }
            for key in self.quantity_variables_names
        }

        self.weight_volume_ratio = None

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

    def _check_data_empty(self) -> bool:
        bs_data = self.bs_data
        siret = self.company_siret

        bs_emitted_data = bs_data[bs_data["emitter_company_siret"] == siret]
        bs_received_data = bs_data[bs_data["recipient_company_siret"] == siret]

        bs_revised_data = self.bs_revised_data

        # If all raw DataFrames are empty, then output data will be empty
        if (len(bs_emitted_data) == len(bs_received_data) == 0) and (
            (bs_revised_data is None) or (len(bs_revised_data) == 0)
        ):
            return True

        # If all values after preprocessing are empty, then output data will be empty
        if all(
            (e == 0) or (e is None) for e in chain(self.emitted_bs_stats.values(), self.received_bs_stats.values())
        ):
            return True

        return False

    def _preprocess_general_statistics(self, bs_emitted_data: pd.DataFrame, bs_received_data: pd.DataFrame) -> None:
        # For incoming and outgoing data, we compute different statistics
        # about the 'bordereaux'.
        # `target` is the destination in each result dictionary
        # where to store the computed value.
        for target, to_process, to_process_packagings in [
            (self.emitted_bs_stats, bs_emitted_data, self.packagings_data),
            (self.received_bs_stats, bs_received_data, self.packagings_data),
        ]:
            df = to_process.copy()

            if self.bs_type == BSFF:
                if (to_process_packagings is None) or (len(to_process_packagings) == 0):
                    # Case when there is BSFFs but no packagings info
                    continue
                df = (
                    to_process[["id", "status", "received_at"]]
                    .merge(
                        to_process_packagings.loc[
                            to_process_packagings["acceptation_status"] == "ACCEPTED",
                            ["bsff_id", "operation_date", "acceptation_weight"],
                        ],
                        left_on="id",
                        right_on="bsff_id",
                        how="left",
                    )
                    .sort_values(
                        "operation_date", ascending=False, na_position="last"
                    )  # Used to capture the date of the last processed packaging or null if there is at least one packaging not processed
                    .groupby("id", as_index=False)
                    .agg(
                        status=pd.NamedAgg(column="status", aggfunc="max"),
                        received_at=pd.NamedAgg(column="received_at", aggfunc="max"),
                        processed_at=pd.NamedAgg(column="operation_date", aggfunc="first"),
                    )
                )

            # total number of 'bordereaux' emitted/received
            target["total"] = len(df)

            # total number of 'bordereaux' that are considered as 'archived' (end of traceability)
            target["archived"] = len(
                df[
                    df["status"].isin(
                        [
                            "PROCESSED",
                            "REFUSED",
                            "NO_TRACEABILITY",
                            "FOLLOWED_WITH_PNTTD",
                            "INTERMEDIATELY_PROCESSED",
                        ]
                    )
                ]
            )

            # DataFrame holding all the 'bordereaux' that have been
            # processed in more than one month.
            bs_emitted_processed_in_more_than_one_month = df[
                ((df["processed_at"] - df["received_at"]) > np.timedelta64(1, "M"))
            ]

            # Total number of bordereaux processed in more than one month
            processed_in_more_than_one_month_count = len(bs_emitted_processed_in_more_than_one_month)

            target["processed_in_more_than_one_month_count"] = processed_in_more_than_one_month_count

            # If there is some 'bordereaux' processed in more than one month,
            # we compute the average processing time.
            if processed_in_more_than_one_month_count:
                res = (
                    (
                        bs_emitted_processed_in_more_than_one_month["processed_at"]
                        - bs_emitted_processed_in_more_than_one_month["received_at"]
                    ).mean()
                ).total_seconds() / (24 * 3600)  # Time in seconds is converted in days
                target["processed_in_more_than_one_month_avg_processing_time"] = f"{format_number_str(res, 1)}j"

            # Handle the case of BSFF specific packagings statistics
            if to_process_packagings is not None:
                # Total number of packagings sent/received
                target["total_packagings"] = len(
                    to_process_packagings[
                        (to_process_packagings["bsff_id"].isin(to_process["id"]))
                        & (~to_process_packagings["operation_date"].isnull())
                    ]
                )

                # Merging of BSFF 'bordereaux' data with associated packagings data
                # as we will need the date of reception that is stored at the 'bordereau' level.
                bs_data_with_packagings = to_process.merge(
                    to_process_packagings,
                    left_on="id",
                    right_on="bsff_id",
                    validate="one_to_many",
                    how="left",
                )

                # DataFrame with all BSFF along with packagings data
                # for packagings that have been processed in more than one month
                bs_data_with_packagings_processed_in_more_than_one_month = bs_data_with_packagings[
                    (bs_data_with_packagings["operation_date"] - bs_data_with_packagings["received_at"])
                    > np.timedelta64(1, "M")
                ]

                # Number of packagings processed in more than one month.
                target["processed_in_more_than_one_month_packagings_count"] = len(
                    bs_data_with_packagings_processed_in_more_than_one_month
                )

                # Average processing times for the packagings processed in more than one month
                res = (
                    (
                        bs_data_with_packagings_processed_in_more_than_one_month["operation_date"]
                        - bs_data_with_packagings_processed_in_more_than_one_month["received_at"]
                    ).mean()
                ).total_seconds() / (24 * 3600)  # Conversion between number of seconds and days
                if not pd.isna(res):
                    target["processed_in_more_than_one_month_packagings_avg_processing_time"] = f"{res:.1f}j"

        # In case there is any 'bordereaux' revision data, we compute
        # the number of 'bordereaux' that have been revised.
        # NOTE: only revision asked by the current organization are computed.
        bs_revised_data = self.bs_revised_data
        if bs_revised_data is not None:
            bs_revised_data = bs_revised_data.copy()
            bs_ids = pd.concat([bs_emitted_data["id"], bs_received_data["id"]])
            bs_revised_data = bs_revised_data[bs_revised_data["bs_id"].isin(bs_ids)]

            self.pending_revisions_count = bs_revised_data[bs_revised_data["status"] == "PENDING"]["id"].nunique()
            self.revised_bs_count = bs_revised_data[bs_revised_data.status == "ACCEPTED"]["bs_id"].nunique()

    def _preprocess_quantities_stats(self, bs_emitted_data: pd.DataFrame, bs_received_data: pd.DataFrame) -> None:
        # We iterate over the different variables chosen to compute the statistics
        for key in self.quantities_stats.keys():
            # If there is a packagings_data DataFrame, then it means that we are
            # computing BSFF statistics, in this case we use the packagings data instead of
            # 'bordereaux' data as quantity information is stored at packaging level
            if self.bs_type == BSFF:
                if self.packagings_data is None:
                    # Case when there is BSFFs but no packagings info
                    continue

                total_quantity_incoming = bs_received_data.merge(
                    self.packagings_data, left_on="id", right_on="bsff_id"
                )[key].sum()
                total_quantity_outgoing = bs_emitted_data.merge(
                    self.packagings_data, left_on="id", right_on="bsff_id"
                )[key].sum()
            else:
                df_received = bs_received_data.copy()
                df_emitted = bs_emitted_data.copy()
                if self.bs_type in [BSDD, BSDD_NON_DANGEROUS, BSDASRI]:
                    # Handle quantity refused
                    df_received["quantity_received"] = df_received["quantity_received"] - df_received[
                        "quantity_refused"
                    ].fillna(0)
                    df_emitted["quantity_received"] = df_emitted["quantity_received"] - df_emitted[
                        "quantity_refused"
                    ].fillna(0)
                total_quantity_incoming = df_received[key].sum()
                total_quantity_outgoing = df_emitted[key].sum()

            self.quantities_stats[key]["total_quantity_incoming"] = total_quantity_incoming
            self.quantities_stats[key]["total_quantity_outgoing"] = total_quantity_outgoing

            incoming_bar_size = 0
            outgoing_bar_size = 0

            if not (total_quantity_incoming == total_quantity_outgoing == 0):
                # The bar sizes are relative to the largest quantity.
                # Size is expressed as percentage of the component width.
                if total_quantity_incoming > total_quantity_outgoing:
                    incoming_bar_size = 100
                    outgoing_bar_size = int(100 * (total_quantity_outgoing / total_quantity_incoming))
                else:
                    incoming_bar_size = int(100 * (total_quantity_incoming / total_quantity_outgoing))
                    outgoing_bar_size = 100
            self.quantities_stats[key]["bar_size_incoming"] = incoming_bar_size
            self.quantities_stats[key]["bar_size_outgoing"] = outgoing_bar_size

        # If both "quantity_received" and "volume" variables have been chosen,
        # then it means that we are computing BSDASRI statistics.
        # In this case we compute the ratio between volume and weight.
        if all(key in self.quantity_variables_names for key in ["quantity_received", "volume"]):
            if (self.quantities_stats["volume"]["total_quantity_incoming"]) > 0:
                self.weight_volume_ratio = (
                    self.quantities_stats["quantity_received"]["total_quantity_incoming"]
                    / self.quantities_stats["volume"]["total_quantity_incoming"]
                )

    def _preprocess_data(self) -> None:
        bs_data = self.bs_data.copy()

        bs_emitted_data = bs_data[
            (bs_data["emitter_company_siret"] == self.company_siret)
            & bs_data["sent_at"].between(*self.data_date_interval)
        ]
        bs_received_data = bs_data[
            (bs_data["recipient_company_siret"] == self.company_siret)
            & bs_data["received_at"].between(*self.data_date_interval)
        ]

        self._preprocess_general_statistics(bs_emitted_data, bs_received_data)

        self._preprocess_quantities_stats(bs_emitted_data, bs_received_data)

    def build_context(self):
        # We use the format_number_str only on variables that holds
        # quantity values.
        ctx = {
            "emitted_bs_stats": {
                k: (format_number_str(v, 0) if isinstance(v, numbers.Number) else v)
                for k, v in self.emitted_bs_stats.items()
            },
            "received_bs_stats": {
                k: (format_number_str(v, 0) if isinstance(v, numbers.Number) else v)
                for k, v in self.received_bs_stats.items()
            },
            "pending_revisions_count": format_number_str(self.pending_revisions_count, precision=0),
            "revised_bs_count": format_number_str(self.revised_bs_count, precision=0),
            # quantities_stats is two level deep so we need to use a nested
            # dict comprehension loop.
            "quantities_stats": {
                ok: {
                    k: (format_number_str(v, 2) if k in ["total_quantity_incoming", "total_quantity_outgoing"] else v)
                    for k, v in ov.items()
                }
                for ok, ov in self.quantities_stats.items()
            },
            # We multiply the weight to get kilograms instead of tons for the weight_volume_ratio
            "weight_volume_ratio": format_number_str(self.weight_volume_ratio * 1000, 2)
            if self.weight_volume_ratio is not None
            else None,
        }

        return ctx

    def build(self):
        self._preprocess_data()

        data = {}
        if not self._check_data_empty():
            data = self.build_context()
        return data


class WasteFlowsTableProcessor:
    """Component that displays an exhaustive tables with input and output wastes classified by waste codes.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    transporters_data_df : Dict[str, pd.DataFrame]
        Dictionary that contains DataFrames related to transporters. Each key in the "Bordereau type" (BSDD, BSDA...)
        and the corresponding value is a pandas DataFrame containing information about the transported waste.
    rndts_data: dict of DataFrames
        DataFrame containing RNDTS statements data.
    data_date_interval : tuple[datetime, datetime]
        Represents the date range for which the data is being processed.
        It consists of two `datetime` objects, the start date and the end date.
    waste_codes_df: DataFrame
        DataFrame containing list of waste codes with their descriptions. It is the waste nomenclature.
    packagings_data : pd.DataFrame | None
        Optional parameter that represents a DataFrame containing data about BSFF packagings.
    """

    def __init__(
        self,
        company_siret: str,
        bs_data_dfs: Dict[str, pd.DataFrame],
        transporters_data_df: Dict[str, pd.DataFrame],  # Handling new multi-modal Trackdéchets feature
        rndts_data: dict[str, pd.DataFrame | None],
        data_date_interval: tuple[datetime, datetime],
        waste_codes_df: pd.DataFrame,
        packagings_data: pd.DataFrame | None = None,
    ) -> None:
        self.bs_data_dfs = bs_data_dfs
        self.transporters_data_df = transporters_data_df
        self.rndts_data = rndts_data
        self.data_date_interval = data_date_interval
        self.waste_codes_df = waste_codes_df
        self.packagings_data = packagings_data
        self.company_siret = company_siret

        self.preprocessed_df = None

    def _preprocess_bs_data(self) -> pd.DataFrame | None:
        siret = self.company_siret

        dfs_to_concat = []
        for bs_type, df in self.bs_data_dfs.items():
            if df is None:
                continue

            df = df.copy()

            # Handling multimodal
            if bs_type in BS_TYPES_WITH_MULTIMODAL_TRANSPORT:
                transport_df = self.transporters_data_df.get(bs_type)

                if transport_df is not None:
                    transport_df = transport_df.copy()
                    if len(df) > 0:
                        df = df.drop(
                            columns=["sent_at"],
                            errors="ignore",
                        )  # To avoid column duplication with transport data

                        if bs_type == BSFF:
                            if self.packagings_data is not None:
                                # Quantity is taken from packagings data in case of BSFF
                                df = df.drop(
                                    columns=["quantity_received"],
                                    errors="ignore",
                                )
                                df = df.merge(
                                    self.packagings_data[["bsff_id", "acceptation_weight", "acceptation_date"]],
                                    left_on="id",
                                    right_on="bsff_id",
                                    validate="one_to_many",
                                )
                                df = df.rename(columns={"acceptation_weight": "quantity_received"})

                                # data is re-aggregated at 'bordereau' granularity to match other 'obordereaux' dfs granularity
                                df = df.groupby("id", as_index=False).agg(
                                    {
                                        "emitter_company_siret": "max",
                                        "recipient_company_siret": "max",
                                        "received_at": "min",
                                        "waste_code": "max",
                                        "quantity_received": "sum",
                                    }
                                )
                            else:
                                # If there is no packagings data, we can't get the quantity
                                continue

                        transport_columns_to_take = ["bs_id", "sent_at", "transporter_company_siret"]

                        validation = "many_to_many"  # Due to merging with packaging before
                        if (not bs_type == BSFF) or (
                            self.packagings_data is None
                        ):  # BSFF stores quantity in packagings data
                            validation = "one_to_many"

                        if bs_type in [BSDD, BSDD_NON_DANGEROUS]:
                            # Handle quantity refused
                            df["quantity_received"] = df["quantity_received"] - df["quantity_refused"].fillna(0)

                        df = df.merge(
                            transport_df[transport_columns_to_take],
                            left_on="id",
                            right_on="bs_id",
                            how="left",
                            validate=validation,
                        )

                        df = df.groupby("id", as_index=False).agg(
                            {
                                "emitter_company_siret": "max",
                                "recipient_company_siret": "max",
                                "transporter_company_siret": lambda x: x.fillna(
                                    value=""
                                ).max(),  # Handle missing values being NA instead of None
                                "sent_at": "min",
                                "received_at": "min",
                                "waste_code": "max",
                                "quantity_received": "max",
                            }
                        )
                    else:
                        df = transport_df
            elif bs_type == "BSDASRI":
                df["quantity_received"] = df["quantity_received"] - df["quantity_refused"].fillna(0)
            dfs_to_concat.append(df)

        if len(dfs_to_concat) == 0:
            return

        df = pd.concat(dfs_to_concat)

        # We create a column to differentiate incoming waste from
        # outgoing and transported waste.
        df["flow_status"] = pd.NA

        # We determine each "flow type", a 'bordereau' can have several flow status (e.g a company that emit and also transport)
        dfs_to_concat = []
        for siret_key, date_key, flow_type in [
            ("emitter_company_siret", "sent_at", "outgoing"),
            ("recipient_company_siret", "received_at", "incoming"),
            ("transporter_company_siret", "sent_at", "transported"),
        ]:
            if (siret_key in df.columns) and (date_key in df.columns):
                temp_df = df[(df[siret_key] == siret) & df[date_key].between(*self.data_date_interval)].copy()
                temp_df["flow_status"] = flow_type
                dfs_to_concat.append(temp_df)

        df = pd.concat(dfs_to_concat)
        df = df.dropna(subset="flow_status")

        if len(df) > 0:
            # We compute the quantity by waste codes and incoming/outgoing/transported categories
            df_grouped = df.groupby(["waste_code", "flow_status"], as_index=False).agg({"quantity_received": "sum"})
            df_grouped["unit"] = "t"
            df_grouped["quantity_received"] = df_grouped["quantity_received"].astype(float).round(3)
            return df_grouped

        return None

    def _preprocess_rndts_data(self) -> pd.DataFrame | None:
        # If there is RNDTS data, we add it to the dataframe
        df_to_group = []
        for key, date_col, siret_col in [
            ("ndw_incoming", "date_reception", "etablissement_numero_identification"),
            ("ndw_outgoing", "date_expedition", "producteur_numero_identification"),
            ("excavated_land_incoming", "date_reception", "etablissement_numero_identification"),
            ("excavated_land_outgoing", "date_expedition", "producteur_numero_identification"),
        ]:
            df_rndts = self.rndts_data.get(key)

            if (df_rndts is not None) and (len(df_rndts) > 0):
                df_rndts = df_rndts.rename(
                    columns={"quantite": "quantity_received", "code_dechet": "waste_code", "unite": "unit"}
                )

                df_rndts["unit"] = df_rndts["unit"].replace({"T": "t", "M3": "m³"})
                # Handle missing waste code for some excavated land statements
                if "excavated_land" in key:
                    df_rndts["waste_code"].fillna("17 05 04")

                rndts_grouped_data = (
                    df_rndts[
                        df_rndts[date_col].between(*self.data_date_interval)
                        & (df_rndts[siret_col] == self.company_siret)
                    ]
                    .groupby(["waste_code", "unit"], as_index=False)["quantity_received"]
                    .sum()
                )  # We group also by unit to account for some wastes quantities that are measured in m³

                if len(rndts_grouped_data) > 0:
                    rndts_grouped_data["flow_status"] = "incoming" if (date_col == "date_reception") else "outgoing"
                    df_to_group.append(rndts_grouped_data)

                # Transport data
                rndts_grouped_data = (
                    df_rndts[
                        df_rndts[date_col].between(*self.data_date_interval)
                        & (df_rndts["numeros_indentification_transporteurs"].apply(lambda x: self.company_siret in x))
                    ]
                    .groupby(["waste_code", "unit"], as_index=False)["quantity_received"]
                    .sum()
                )

                if len(rndts_grouped_data) > 0:
                    rndts_grouped_data["flow_status"] = (
                        "transported_incoming" if (date_col == "date_reception") else "transported_outgoing"
                    )
                    df_to_group.append(rndts_grouped_data)

        res = None
        if len(df_to_group) > 0:
            res = pd.concat(df_to_group)
            res = res.reset_index(drop=True)

        return res

    def _preprocess_data(self):
        bs_grouped_data = self._preprocess_bs_data()
        rndts_grouped_data = self._preprocess_rndts_data()

        df_grouped = pd.DataFrame()
        match (bs_grouped_data, rndts_grouped_data):
            case (None, None):
                return
            case (pd.DataFrame(), pd.DataFrame()):
                df_grouped = pd.concat([bs_grouped_data, rndts_grouped_data])
            case (df, None) | (None, df):
                df_grouped = df
            case _:
                raise ValueError()

        # We add the waste code description from the waste nomenclature
        final_df = pd.merge(
            df_grouped,
            self.waste_codes_df,
            left_on="waste_code",
            right_index=True,
            how="left",
            validate="many_to_one",
        )

        final_df = final_df[final_df["quantity_received"] > 0]
        final_df["quantity_received"] = final_df["quantity_received"].apply(lambda x: format_number_str(x, 3))
        final_df["description"] = final_df["description"].fillna("")
        final_df = (
            final_df[["waste_code", "description", "flow_status", "quantity_received", "unit"]]
            .sort_values(by=["waste_code", "flow_status", "unit"], ascending=[True, True, False])
            .reset_index(drop=True)
        )

        self.preprocessed_df = final_df

    def _check_empty_data(self) -> bool:
        if self.preprocessed_df is None:
            return True

        return False

    def build_context(self):
        return self.preprocessed_df.to_dict("records")

    def build(self):
        self._preprocess_data()

        res = {}

        if not self._check_empty_data():
            res = self.build_context()
        return res


class BsdCanceledTableProcessor:
    """Component that displays an exhaustive tables with the list of 'bordereaux' that have been canceled.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    bs_revised_data: DataFrame
        Dict with key being the 'bordereau' type and values the DataFrame containing the associated revision data.
    data_date_interval : tuple[datetime, datetime]
        Represents the date range for which the data is being processed.
        It consists of two `datetime` objects, the start date and the end date.
    """

    def __init__(
        self,
        company_siret: str,
        bs_data_dfs: Dict[str, pd.DataFrame],
        bs_revised_data: Dict[str, pd.DataFrame],
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.bs_data_dfs = bs_data_dfs
        self.data_date_interval = data_date_interval
        self.bs_revised_data = bs_revised_data
        self.company_siret = company_siret

        self.preprocessed_df = pd.DataFrame()

    def _preprocess_data(self) -> None:
        if not self.bs_revised_data:
            return

        dfs = []
        for bs_type, revised_data_df in self.bs_revised_data.items():
            # Cancellation events are stored in revisions
            cancellations = revised_data_df[
                revised_data_df.is_canceled & revised_data_df.updated_at.between(*self.data_date_interval)
            ]
            if len(cancellations):
                bs_data = self.bs_data_dfs[bs_type]

                # Columns that will be displayed in the output table
                columns_to_take = [
                    "id_y",  # Will correspond to the 'bordereau' id after merge
                    "quantity_received",
                    "emitter_company_siret",
                    "recipient_company_siret",
                    "waste_code",
                    "updated_at",
                    "comment",
                ]

                # Human-friendly id is stored in the readable_id column in the case of BSDDs
                if "readable_id" in bs_data.columns:
                    columns_to_take.append("readable_id")

                # BSDASRI does not have waste name
                if "waste_name" in bs_data.columns:
                    columns_to_take.append("waste_name")

                # Handle quantity refused
                if "quantity_refused" in bs_data.columns:
                    columns_to_take.append("quantity_refused")

                temp_df = pd.merge(
                    cancellations,
                    bs_data,
                    left_on="bs_id",
                    right_on="id",
                ).sort_values("updated_at")

                temp_df = temp_df[columns_to_take]
                temp_df = temp_df.rename(columns={"id_y": "id"})

                dfs.append(temp_df)

        if dfs:
            self.preprocessed_df = pd.concat(dfs)

    def _check_empty_data(self) -> bool:
        if len(self.preprocessed_df) == 0:
            return True

        return False

    def build_context(self):
        data = self.preprocessed_df
        data["updated_at"] = data["updated_at"].dt.strftime("%d/%m/%Y %H:%M")
        return json.loads(data.to_json(orient="records"))

    def build(self):
        self._preprocess_data()

        data = {}
        if not self._check_empty_data():
            data = self.build_context()

        return data


class SameEmitterRecipientTableProcessor:
    """Component that displays an exhaustive tables with the
    list of 'bordereaux' that have the same company
    as emitter and recipient along with a worksite address.

    Parameters
    ----------
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    transporters_data_df : Dict[str, pd.DataFrame]
        Dictionary that contains DataFrames related to transporters. Each key in the "Bordereau type" (BSDD, BSDA...)
        and the corresponding value is a pandas DataFrame containing information about the transported waste.
    data_date_interval : tuple[datetime, datetime]
        Represents the date range for which the data is being processed.
        It consists of two `datetime` objects, the start date and the end date.
    """

    def __init__(
        self,
        bs_data_dfs: Dict[str, pd.DataFrame],
        transporters_data_dfs: Dict[str, pd.DataFrame],
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.bs_data_dfs = bs_data_dfs
        self.transporters_data_dfs = transporters_data_dfs
        self.data_date_interval = data_date_interval

        self.preprocessed_df = pd.DataFrame()

    def _preprocess_data(self) -> None:
        # This case only works on BSDD and BSDA so we filter others type of "bordereaux"
        dfs_to_process = {
            bs_type: df.copy()
            for bs_type, df in self.bs_data_dfs.items()
            if bs_type in [BSDD, BSDD_NON_DANGEROUS, BSDA]
        }

        columns_to_take = [
            "id",
            "readable_id",
            "sent_at",
            "received_at",
            "quantity_received",
            "waste_code",
            "waste_name",
            "worksite_name",
            "worksite_address",
            "emitter_company_siret",
            "recipient_company_siret",
        ]
        dfs_processed = []

        for bs_type, df in dfs_to_process.items():
            transport_df = self.transporters_data_dfs.get(bs_type)

            if (df is None) or (transport_df is None):
                continue

            columns_to_drop = ["sent_at", "transporter_company_siret"]

            # Handling multimodal
            df.drop(
                columns=columns_to_drop,
                errors="ignore",
                inplace=True,
            )  # To avoid column duplication with transport data

            transport_df_columns_to_take = ["bs_id", "sent_at", "transporter_company_siret"]

            df = df.merge(
                transport_df[transport_df_columns_to_take],
                left_on="id",
                right_on="bs_id",
                how="left",
                validate="one_to_many",
            )

            df = df.groupby("id", as_index=False).agg(
                {c: ("min" if c in ["sent_at", "received_at"] else "max") for c in columns_to_take if c in df.columns}
            )
            same_emitter_recipient_df = df[
                (df["emitter_company_siret"] == df["recipient_company_siret"])
                & df["worksite_address"].notna()
                & df["sent_at"].between(*self.data_date_interval)
            ].reindex(columns_to_take, axis=1)
            if len(same_emitter_recipient_df):
                dfs_processed.append(same_emitter_recipient_df)

        if dfs_processed:
            self.preprocessed_df = pd.concat(dfs_processed)

    def _check_empty_data(self) -> bool:
        if len(self.preprocessed_df) == 0:
            return True

        return False

    def build_context(self):
        data = self.preprocessed_df
        data["sent_at"] = pd.to_datetime(data["sent_at"]).dt.strftime("%d/%m/%Y %H:%M")
        data["received_at"] = pd.to_datetime(data["received_at"]).dt.strftime("%d/%m/%Y %H:%M")

        return json.loads(data.to_json(orient="records"))

    def build(self):
        self._preprocess_data()

        data = {}
        if not self._check_empty_data():
            data = self.build_context()

        return data


class StorageStatsProcessor:
    """Component that displays waste stock on site by waste codes (TOP 4) and total stock in tons.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    waste_codes_df: DataFrame
        DataFrame containing list of waste codes with their descriptions.
    data_date_interval : tuple[datetime, datetime]
        Represents the date range for which the data is being processed.
        It consists of two `datetime` objects, the start date and the end date.
    """

    def __init__(
        self,
        company_siret: str,
        bs_data_dfs: Dict[str, pd.DataFrame],
        waste_codes_df: pd.DataFrame,
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret

        self.bs_data_dfs = bs_data_dfs
        self.waste_codes_df = waste_codes_df
        self.data_date_interval = data_date_interval

        self.stock_by_waste_code = None
        self.total_stock = None

    def _preprocess_data(self) -> pd.Series | None:
        siret = self.company_siret

        dfs_to_concat = [df for bs_type, df in self.bs_data_dfs.items() if bs_type != BSDD_NON_DANGEROUS]

        if len(dfs_to_concat) == 0:
            self.stock_by_waste_code = pd.Series()
            return

        df = pd.concat(dfs_to_concat)

        # Handle quantity refused
        if "quantity_refused" in df.columns:
            df["quantity_received"] = df["quantity_received"] - df["quantity_refused"].fillna(0)

        emitted_mask = (df.emitter_company_siret == siret) & df.sent_at.between(*self.data_date_interval)
        received_mask = (df.recipient_company_siret == siret) & df.received_at.between(*self.data_date_interval)

        emitted = df[emitted_mask].groupby("waste_code")["quantity_received"].sum()
        received = df[received_mask].groupby("waste_code")["quantity_received"].sum()

        # Index wise sum (index being the waste codes)
        # to compute the theoretical stock of waste
        # (difference between incoming and outgoing quantities)
        stock_by_waste_code: pd.Series = (-emitted + received).fillna(-emitted).fillna(received)
        stock_by_waste_code.sort_values(ascending=False, inplace=True)

        # Only positive differences are kept
        stock_by_waste_code = stock_by_waste_code[stock_by_waste_code > 0]
        total_stock = format_number_str(stock_by_waste_code.sum(), precision=1)
        stock_by_waste_code = stock_by_waste_code.apply(format_number_str, precision=1)

        # Data is enriched with waste description from the waste nomenclature
        stock_by_waste_code = pd.merge(
            stock_by_waste_code,
            self.waste_codes_df,
            left_index=True,
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        stock_by_waste_code["description"].fillna("", inplace=True)

        self.stock_by_waste_code = stock_by_waste_code
        self.total_stock = total_stock

    def _check_data_empty(self) -> bool:
        if (len(self.stock_by_waste_code) == 0) or self.stock_by_waste_code["quantity_received"].isna().all():
            return True

        return False

    def _add_stats(self):
        stored_waste = []

        for row in self.stock_by_waste_code.head(4).itertuples():
            stored_waste.append(
                {
                    "quantity_received": row.quantity_received,
                    "code": str(row.Index),
                    "description": row.description,
                }
            )
        return {"stored_waste": stored_waste, "total_stock": self.total_stock}

    def build(self):
        self._preprocess_data()

        data = {}
        if not self._check_data_empty():
            data = self._add_stats()
        return data


class ICPEItemsProcessor:
    """Component that displays list of ICPE authorized items.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    icpe_data: DataFrame
        DataFrame containing list of ICPE authorized items
    """

    def __init__(
        self,
        company_siret: str,
        icpe_data: pd.DataFrame | None,
    ) -> None:
        self.company_siret = company_siret
        self.icpe_data = icpe_data

        self.preprocessed_df = None

    def _preprocess_data(self):
        df = self.icpe_data

        if df is None:
            return

        df["quantite"] = df["quantite"].apply(format_number_str, precision=3)

        df = df.sort_values(["rubrique"])

        self.preprocessed_df = df

    def build_context(self) -> dict:
        data = self.preprocessed_df

        # Handle "nan" textual values not being converted to JSON null
        data["quantite"] = data["quantite"].replace("nan", pd.NA)
        return json.loads(data.to_json(orient="records"))

    def _check_empty_data(self) -> bool:
        if self.preprocessed_df is None or len(self.preprocessed_df) == 0:
            return True

        return False

    def build(self):
        self._preprocess_data()

        data = {}
        if not self._check_empty_data():
            data = self.build_context()

        return data


class TraceabilityInterruptionsProcessor:
    """Component that displays list of declared traceability interruptions.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bsdd_data: DataFrame
        DataFrame containing BSDD data.
    waste_codes_df: DataFrame
        DataFrame containing list of waste codes with their descriptions.
    data_date_interval : tuple[datetime, datetime]
        Represents the date range for which the data is being processed.
        It consists of two `datetime` objects, the start date and the end date.
    """

    def __init__(
        self,
        company_siret: str,
        bsdd_data: pd.DataFrame,
        waste_codes_df: pd.DataFrame,
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret

        self.bsdd_data = bsdd_data
        self.waste_codes_df = waste_codes_df
        self.data_date_interval = data_date_interval

        self.preprocessed_data = None

    def _preprocess_data(self) -> None:
        if self.bsdd_data is None:
            return

        df_filtered = self.bsdd_data[
            self.bsdd_data["no_traceability"]
            & (self.bsdd_data["recipient_company_siret"] == self.company_siret)
            & self.bsdd_data["received_at"].between(*self.data_date_interval)
        ].copy()

        if len(df_filtered) == 0:
            return

        # Handle quantity refused
        df_filtered["quantity_received"] = df_filtered["quantity_received"] - df_filtered["quantity_refused"].fillna(0)

        # Quantity and count are computed by waste code
        df_grouped = df_filtered.groupby("waste_code", as_index=False).agg(
            quantity=pd.NamedAgg(column="quantity_received", aggfunc="sum"),
            count=pd.NamedAgg(column="id", aggfunc="count"),
        )

        # Data is enriched with waste description from the waste nomenclature
        final_df = pd.merge(
            df_grouped,
            self.waste_codes_df,
            left_on="waste_code",
            right_index=True,
            how="left",
            validate="one_to_one",
        )

        final_df["quantity"] = final_df["quantity"].apply(format_number_str, precision=2)

        self.preprocessed_data = final_df

    def _check_data_empty(self) -> bool:
        if (self.preprocessed_data is None) or (len(self.preprocessed_data) == 0):
            return True

        return False

    def _add_stats(self) -> list:
        stats = []

        for e in self.preprocessed_data.sort_values("quantity", ascending=False).itertuples():
            row = {
                "waste_code": e.waste_code,
                "count": e.count,
                "quantity": e.quantity,
                "description": e.description,
            }
            stats.append(row)
        return stats

    def build(self) -> list:
        self._preprocess_data()

        if not self._check_data_empty():
            return self._add_stats()
        return []


class WasteIsDangerousStatementsProcessor:
    """Component that displays list of wastes tracked
    without dangerous waste code but with the "waste_is_dangerous" option enabled.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bsdd_data: DataFrame
        DataFrame containing bsdd data.
    bsdd_transporters_data : DataFrame
        DataFrames containing information about the transported BSDD waste.
    waste_codes_df: DataFrame
        DataFrame containing list of waste codes with their descriptions.
    data_date_interval : tuple[datetime, datetime]
        Represents the date range for which the data is being processed.
        It consists of two `datetime` objects, the start date and the end date.
    """

    def __init__(
        self,
        company_siret: str,
        bsdd_data: pd.DataFrame,
        bsdd_transporters_data: pd.DataFrame | None,
        waste_codes_df: pd.DataFrame,
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret

        self.bsdd_data = bsdd_data
        self.bsdd_transporters_data = bsdd_transporters_data
        self.waste_codes_df = waste_codes_df
        self.data_date_interval = data_date_interval

        self.preprocessed_data = None

    def _preprocess_data(self) -> None:
        if (self.bsdd_data is None) or (self.bsdd_transporters_data is None):
            return

        df = self.bsdd_data.copy()
        transport_df = self.bsdd_transporters_data

        # Handling multimodal
        df.drop(
            columns=["sent_at"],
            errors="ignore",
            inplace=True,
        )  # To avoid column duplication with transport data

        df = df.merge(
            transport_df[["bs_id", "sent_at", "transporter_company_siret"]],
            left_on="id",
            right_on="bs_id",
            how="left",
            validate="one_to_many",
        )

        df = df.groupby("id", as_index=False).agg(
            {
                "is_dangerous": "max",
                "emitter_company_siret": "max",
                "waste_code": "max",
                "sent_at": "min",
                "quantity_received": "max",
                "quantity_refused": "max",
            }
        )

        df_filtered = df[
            df["is_dangerous"]
            & (df["emitter_company_siret"] == self.company_siret)
            & (~df["waste_code"].str.contains(pat=r".*\*$"))
            & (df["sent_at"].between(*self.data_date_interval))
        ]

        if len(df_filtered) == 0:
            return

        # Handle quantity refused
        df_filtered["quantity_received"] = df_filtered["quantity_received"] - df_filtered["quantity_refused"].fillna(0)

        df_grouped = df_filtered.groupby("waste_code", as_index=False).agg(
            quantity=pd.NamedAgg(column="quantity_received", aggfunc="sum"),
            count=pd.NamedAgg(column="id", aggfunc="count"),
        )

        final_df = pd.merge(
            df_grouped,
            self.waste_codes_df,
            left_on="waste_code",
            right_index=True,
            how="left",
            validate="one_to_one",
        )

        final_df["quantity"] = final_df["quantity"].apply(format_number_str, precision=2)

        self.preprocessed_data = final_df

    def _check_data_empty(self) -> bool:
        if (self.preprocessed_data is None) or (len(self.preprocessed_data) == 0):
            return True

        return False

    def _add_stats(self) -> list:
        stats = []

        for e in self.preprocessed_data.sort_values("quantity", ascending=False).itertuples():
            row = {
                "waste_code": e.waste_code,
                "count": e.count,
                "quantity": e.quantity,
                "description": e.description,
            }
            stats.append(row)
        return stats

    def build(self) -> list:
        self._preprocess_data()

        if not self._check_data_empty():
            return self._add_stats()
        return []


class ReceiptAgrementsProcessor:
    """Component that displays informations about the company receipts and agreements.

    Parameters
    ----------
    receipts_agreements_data : dict
        Dict with keys being the name of the receipt/agreement and values being DataFrames
        with one line per receipt/agreement (usually there is only one receipt for a receipt type for an establishment but
        there might be more).
    """

    def __init__(self, receipts_agreements_data: Dict[str, pd.DataFrame]) -> None:
        self.receipts_agreements_data = receipts_agreements_data

    def _check_data_empty(self) -> bool:
        if len(self.receipts_agreements_data) == 0:
            return True

        return False

    def build(self):
        res = []
        for name, data in self.receipts_agreements_data.items():
            for line in data.itertuples():
                validity_str = ""
                if "validity_limit" in line._fields:
                    # todo: utcnow
                    if line.validity_limit < datetime.now():
                        validity_str = f"expiré depuis le {line.validity_limit:%d/%m/%Y}"
                    else:
                        validity_str = f"valide jusqu'au {line.validity_limit:%d/%m/%Y}"
                res.append(
                    {
                        "name": name,
                        "number": line.receipt_number,
                        "validity_str": validity_str,
                    }
                )

        return res


class PrivateIndividualsCollectionsTableProcessor:
    """Component that displays a list of private individuals where waste have been picked up.
    Only for BSDA data.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bsda_data_df: DataFrame
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    bsda_transporters_data_df : DataFrame
        DataFrames containing information about the transported BSDA waste.
    data_date_interval : tuple[datetime, datetime]
        Represents the date range for which the data is being processed.
        It consists of two `datetime` objects, the start date and the end date.
    """

    def __init__(
        self,
        company_siret: str,
        bsda_data_df: pd.DataFrame,
        bsda_transporters_data_df: pd.DataFrame | None,
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.bsda_transporters_data_df = bsda_transporters_data_df
        self.bsda_data_df = bsda_data_df
        self.data_date_interval = data_date_interval

        self.preprocessed_data = None

    def _preprocess_data(self) -> None:
        df = self.bsda_data_df.copy()
        transport_df = self.bsda_transporters_data_df

        if (df is None) or (transport_df is None):
            return

        # Handling multimodal
        df.drop(
            columns=["sent_at"],
            errors="ignore",
            inplace=True,
        )  # To avoid column duplication with transport data

        df = df.merge(
            transport_df[["bs_id", "sent_at", "transporter_company_siret"]],
            left_on="id",
            right_on="bs_id",
            how="left",
            validate="one_to_many",
        )

        df = df.groupby("id", as_index=False).agg(
            {
                "emitter_is_private_individual": "max",
                "recipient_company_siret": "max",
                "worker_company_siret": "max",
                "emitter_company_name": "max",
                "emitter_company_address": "max",
                "worksite_name": "max",
                "worksite_address": "max",
                "waste_code": "max",
                "waste_name": "max",
                "quantity_received": "max",
                "sent_at": "min",
                "received_at": "min",
            }
        )

        filtered_df = df[
            (
                (df["recipient_company_siret"] == self.company_siret)
                | (df["worker_company_siret"] == self.company_siret)
            )
            & df["emitter_is_private_individual"]
            & df["sent_at"].between(*self.data_date_interval)
        ]

        if len(filtered_df) > 0:
            self.preprocessed_data = filtered_df

    def _check_data_empty(self) -> bool:
        if self.preprocessed_data is None:
            return True

        return False

    def _add_stats(self) -> list:
        stats = []

        for e in self.preprocessed_data.sort_values("sent_at").itertuples():
            row = {
                "id": e.id,
                "recipient_company_siret": e.recipient_company_siret,
                "worker_company_siret": e.worker_company_siret if not pd.isna(e.worker_company_siret) else None,
                "emitter_company_name": e.emitter_company_name,
                "emitter_company_address": e.emitter_company_address,
                "worksite_name": e.worksite_name if not pd.isna(e.worksite_name) else None,
                "worksite_address": e.worksite_address if not pd.isna(e.worksite_address) else None,
                "waste_code": e.waste_code,
                "waste_name": e.waste_name,
                "quantity": e.quantity_received if not pd.isna(e.quantity_received) else None,
                "sent_at": e.sent_at.strftime("%d/%m/%Y %H:%M") if not pd.isna(e.sent_at) else None,
                "received_at": e.received_at.strftime("%d/%m/%Y %H:%M") if not pd.isna(e.received_at) else None,
            }
            stats.append(row)
        return stats

    def build(self) -> list:
        self._preprocess_data()

        if not self._check_data_empty():
            return self._add_stats()
        return []


class QuantityOutliersTableProcessor:
    """Component that displays a list of bordereaux with outliers values on quantity.

    Parameters
    ----------
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    transporters_data_df : Dict[str, pd.DataFrame]
        Dictionary that contains DataFrames related to transporters. Each key in the "Bordereau type" (BSDD, BSDA...)
        and the corresponding value is a pandas DataFrame containing information about the transported waste.
    data_date_interval : tuple[datetime, datetime]
        Represents the date range for which the data is being processed.
        It consists of two `datetime` objects, the start date and the end date.
    packagings_data_df : pd.DataFrame | None
        Optional parameter that represents a DataFrame containing data about BSFF packagings.
    """

    def __init__(
        self,
        bs_data_dfs: Dict[str, pd.DataFrame],
        transporters_data_df: Dict[str, pd.DataFrame],
        data_date_interval: tuple[datetime, datetime],
        packagings_data_df: pd.DataFrame | None = None,
    ) -> None:
        self.bs_data_dfs = bs_data_dfs
        self.transporters_data_df = transporters_data_df
        self.packagings_data_df = packagings_data_df
        self.data_date_interval = data_date_interval

        self.preprocessed_data = None

    def get_quantity_outliers(
        self,
        df: pd.DataFrame,
        bs_type: str,
        transporters_df: pd.DataFrame | None,
        packagings_data_df: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        """Get lines from 'bordereau' DataFrame with inconsistent received quantity.
        The rules to identify outliers in received quantity are business rules and may be tweaked in the future.

        Parameters
        ----------
        df : DataFrame
            DataFrame with 'bordereau' data.
        bs_type : str
            Name of the 'bordereau' (BSDD, BSDD_NON_DANGEROUS, BSDA, BSFF, BSVHU or BSDASRI).

        Returns
        -------
        DataFrame
            DataFrame with lines with received quantity outliers removed.
        """
        df_quantity_outliers = pd.DataFrame()
        if bs_type in [BSDD, BSDD_NON_DANGEROUS, BSDA, BSFF] and (transporters_df is not None):
            # In this case we use transporter data

            # Old 'bordereaux' data could contain sent_at column, we want to use the one from transport data
            df = df.drop(columns=["sent_at"], errors="ignore")

            df_with_transport = df.merge(
                transporters_df[["bs_id", "transporter_transport_mode", "sent_at", "quantity_received"]],
                left_on="id",
                right_on="bs_id",
                how="left",
                validate="one_to_many",
                suffixes=("", "_transport"),
            )

            df_quantity_outliers = df_with_transport[
                (df_with_transport["quantity_received"] > 40)
                & (
                    (df_with_transport["transporter_transport_mode"] == "ROAD")
                    | df_with_transport["transporter_transport_mode"].isna()
                )
                & df_with_transport["sent_at"].between(*self.data_date_interval)
            ].drop_duplicates("id")

        elif bs_type == BSDASRI:
            df_quantity_outliers = df[
                (df["quantity_received"] > 20)
                & (df["transporter_transport_mode"] == "ROAD")
                & (df["sent_at"].between(*self.data_date_interval))
            ]
        elif bs_type == BSVHU:
            df_quantity_outliers = df[
                (df["quantity_received"] > 40) & (df["sent_at"].between(*self.data_date_interval))
            ]
        else:
            #
            return

        df_quantity_outliers["bs_type"] = bs_type if bs_type != BSDD_NON_DANGEROUS else "bsdd"
        return df_quantity_outliers

    def _preprocess_data(self) -> None:
        outliers_dfs = []
        for bs_type, df in self.bs_data_dfs.items():
            if len(df) == 0:
                continue

            packagings_data_df = None
            if bs_type == BSFF:
                packagings_data_df = self.packagings_data_df

            transporters_df = self.transporters_data_df.get(bs_type, None)
            df_outliers = self.get_quantity_outliers(df, bs_type, transporters_df, packagings_data_df)

            if (df_outliers is not None) and (len(df_outliers) != 0):
                if bs_type in [BSDD, BSDD_NON_DANGEROUS]:
                    df_outliers["id"] = df_outliers["readable_id"]

                outliers_dfs.append(df_outliers)

        if outliers_dfs:
            self.preprocessed_data = pd.concat(outliers_dfs)

    def _check_data_empty(self) -> bool:
        if self.preprocessed_data is None:
            return True

        return False

    def _add_stats(self) -> list:
        stats = []

        for e in self.preprocessed_data.sort_values("sent_at").itertuples():
            row = {
                "id": e.id,
                "bs_type": e.bs_type,
                "emitter_company_siret": e.emitter_company_siret,
                "recipient_company_siret": e.recipient_company_siret,
                "waste_code": e.waste_code,
                "waste_name": e.waste_name if e.bs_type != "bsvhu" else None,
                "quantity": format_number_str(e.quantity_received, 1) if not pd.isna(e.quantity_received) else None,
                "quantity_refused": format_number_str(e.quantity_refused, 1)
                if not pd.isna(e.quantity_refused)
                else None,
                "sent_at": e.sent_at.strftime("%d/%m/%Y %H:%M") if not pd.isna(e.sent_at) else None,
                "received_at": e.received_at.strftime("%d/%m/%Y %H:%M") if not pd.isna(e.received_at) else None,
            }
            stats.append(row)
        return stats

    def build(self) -> list:
        self._preprocess_data()

        if not self._check_data_empty():
            return self._add_stats()
        return []


class WasteProcessingWithoutICPERubriqueProcessor:
    """Component that detects when waste is processed without having a 'rubrique' in ICPE data.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    ndts_incoming_data: DataFrame
        DataFrame containing data for incoming non dangerous waste (from RNDTS).
    icpe_data: pd.DataFrame
        DataFrame containing the list of authorized 'rubriques'.
    data_date_interval : tuple[datetime, datetime]
        Represents the date range for which the data is being processed.
        It consists of two `datetime` objects, the start date and the end date.
    packagings_data_df : pd.DataFrame | None
        Optional parameter that represents a DataFrame containing data about BSFF packagings.
    """

    def __init__(
        self,
        company_siret: str,
        bs_data_dfs: Dict[str, pd.DataFrame | None],
        rndts_incoming_data: pd.DataFrame | None,
        icpe_data: pd.DataFrame | None,
        data_date_interval: tuple[datetime, datetime],
        packagings_data_df: pd.DataFrame | None = None,
    ) -> None:
        self.siret = company_siret
        self.bs_data_dfs = bs_data_dfs
        self.rndts_incoming_data = rndts_incoming_data
        self.icpe_data = icpe_data
        self.data_date_interval = data_date_interval
        self.packagings_data_df = packagings_data_df

        self.preprocessed_data = {
            "dangerous": [],
            "non_dangerous": [],
        }

    def _preprocess_data_multi_rubriques(self) -> None:
        has_2760_1 = False
        has_2760_2 = False
        icpe_data = self.icpe_data

        if (icpe_data is not None) and (len(icpe_data) > 0):
            has_2760_1 = len(icpe_data[(icpe_data["rubrique"] == "2760-1")]) > 0
            has_2760_2 = len(icpe_data[icpe_data["rubrique"].isin(["2760-2", "2760-2-a", "2760-2-b"])]) > 0

        if not has_2760_1:  # Means no authorization for ICPE 2760-1
            bs_2760_dfs = []
            bsdd_data = self.bs_data_dfs[BSDD]

            if (bsdd_data is not None) and (len(bsdd_data) > 0):
                bsdd_data_filtered = bsdd_data[
                    (bsdd_data["recipient_company_siret"] == self.siret)
                    & (bsdd_data["processing_operation_code"] == "D5")
                    & (bsdd_data["processed_at"].between(*self.data_date_interval))
                ]

                if len(bsdd_data_filtered):
                    bsdd_data_filtered["bs_type"] = "BSDD"
                    bs_2760_dfs.append(bsdd_data_filtered)

                if not has_2760_2:  # Means no authorization for ICPE 2760-1 NEITHER 2760-2 (BSDA case)
                    bsda_data = self.bs_data_dfs[BSDA]

                    if (bsda_data is not None) and (len(bsda_data) > 0):
                        bsda_data_filtered = bsda_data[
                            (bsda_data["recipient_company_siret"] == self.siret)
                            & (bsda_data["processing_operation_code"] == "D5")
                            & (bsdd_data["processed_at"].between(*self.data_date_interval))
                        ]
                        if len(bsda_data_filtered) > 0:
                            bsda_data_filtered["bs_type"] = "BSDA"
                            bs_2760_dfs.append(bsda_data_filtered)

            if len(bs_2760_dfs) > 0:
                bs_df = pd.concat(bs_2760_dfs).reset_index(drop=True)  # Creates the list of bordereaux
                self.preprocessed_data["dangerous"].append(
                    {
                        "missing_rubriques": "2760-1, 2760-2",
                        "num_missing_rubriques": 2,
                        "found_processing_codes": "D5",
                        "num_found_processing_codes": 1,
                        "bs_list": bs_df,
                        "stats": {
                            "total_bs": format_number_str(len(bs_df), 0),  # Total number of bordereaux
                            "total_quantity": format_number_str(
                                (bs_df["quantity_received"] - bs_df["quantity_refused"].fillna(0)).sum(), 2
                            ),  # Total quantity processed
                        },
                    }
                )

    def _preprocess_data_single_rubrique(self) -> None:
        configs = [
            {
                "rubrique": "2770",
                "data": [
                    (bs_type, df)
                    for bs_type, df in self.bs_data_dfs.items()
                    if bs_type in [BSDD, BSDA, BSFF, BSDASRI, BSVHU]
                ],
                "processing_codes": ["D10", "R1"],
            },
            {
                "rubrique": "2718-1",
                "data": [
                    (bs_type, df) for bs_type, df in self.bs_data_dfs.items() if bs_type in [BSDD, BSDA, BSFF, BSDASRI]
                ],
                "processing_codes": ["D13", "D14", "D15", "R12", "R13", "D9"],
            },
            {
                "rubrique": "2790",
                "data": [
                    (bs_type, df)
                    for bs_type, df in self.bs_data_dfs.items()
                    if bs_type in [BSDD, BSDA, BSFF, BSDASRI, BSVHU]
                ],
                "processing_codes": [
                    "D8",
                    "D9F",
                    "R2",
                    "R3",
                    "R4",
                    "R5",
                    "R6",
                    "R7",
                    "R9",
                ],
            },
        ]

        for config in configs:
            rubrique = config["rubrique"]

            has_rubrique = False
            icpe_data = self.icpe_data
            if (icpe_data is not None) and (len(icpe_data) > 0):
                has_rubrique = len(icpe_data[icpe_data["rubrique"] == rubrique]) > 0

            if not has_rubrique:
                df_to_process = config["data"]

                bs_filtered_df = self._preprocess_and_filter_bs_list(
                    self.siret,
                    df_to_process,
                    config["processing_codes"],
                    self.data_date_interval,
                    self.packagings_data_df,
                )

                if "quantity_refused" in bs_filtered_df.columns:
                    bs_filtered_df["quantity_received"] = bs_filtered_df["quantity_received"] - bs_filtered_df[
                        "quantity_refused"
                    ].fillna(0)

                if len(bs_filtered_df) > 0:
                    found_processing_codes = bs_filtered_df["processing_operation_code"].unique()
                    self.preprocessed_data["dangerous"].append(
                        {
                            "bs_list": bs_filtered_df.reset_index(drop=True),  # Creates the list of bordereaux
                            "missing_rubriques": rubrique,
                            "num_missing_rubriques": 1,
                            "found_processing_codes": ", ".join(found_processing_codes),
                            "num_found_processing_codes": len(found_processing_codes),
                            "stats": {
                                "total_bs": format_number_str(len(bs_filtered_df), 0),  # Total number of bordereaux
                                "total_quantity": format_number_str(
                                    bs_filtered_df["quantity_received"].sum(), 2
                                ),  # Total quantity processed
                            },
                        }
                    )

    def _preprocess_non_dangerous_rubriques(self) -> None:
        rndts_data_df = self.rndts_incoming_data

        if rndts_data_df is None:
            return

        rndts_data_df = rndts_data_df[
            (rndts_data_df["etablissement_numero_identification"] == self.siret)
            & rndts_data_df["date_reception"].between(*self.data_date_interval)
        ]

        if len(rndts_data_df) == 0:
            return

        rubriques_mapping = [
            {
                "rubriques": ["2760-2"],
                "processing_codes": ["D5"],
            },
            {
                "rubriques": ["2771"],
                "processing_codes": [
                    "D10",
                ],
            },
            {
                "rubriques": ["2771", "2791"],
                "processing_codes": [
                    "D9",
                    "R1",
                    "R2",
                    "R5",
                    "R7",
                ],
            },
            {
                "rubriques": ["2791"],
                "processing_codes": [
                    "D8",
                    "R3",
                    "R4",
                    "R8",
                    "R12",
                ],
            },
        ]

        for mapping in rubriques_mapping:
            rubriques = mapping["rubriques"]

            has_rubrique = False
            icpe_data_df = self.icpe_data
            missing_rubriques = rubriques
            if icpe_data_df is not None:
                # Handle 2791 case that can have alinea
                installation_rubriques = (
                    icpe_data_df["rubrique"].str[:6].str.replace(pat="^2791.*", repl="2791").unique()
                )

                # To handle the case of rubriques with trailing "-a" or trailing "-b", we use only the 6 first characters
                missing_rubriques = set(rubriques) - set(installation_rubriques)
                has_rubrique = len(missing_rubriques) == 0

            if has_rubrique:
                continue

            processing_codes = mapping["processing_codes"]

            filtered_rndts_data_df = rndts_data_df[rndts_data_df["code_traitement"].isin(processing_codes)]

            if len(filtered_rndts_data_df) > 0:
                found_processing_codes = filtered_rndts_data_df["code_traitement"].unique().tolist()

                self.preprocessed_data["non_dangerous"].append(
                    {
                        "missing_rubriques": ", ".join(missing_rubriques),
                        "num_missing_rubriques": len(missing_rubriques),
                        "found_processing_codes": ", ".join(found_processing_codes),
                        "num_found_processing_codes": len(found_processing_codes),
                        "statements_list": filtered_rndts_data_df.sort_values("date_reception")
                        .reset_index(drop=True)
                        .rename(
                            columns={"etablissement_numero_identification": "siret"}
                        ),  # Creates the list of statements
                        "stats": {
                            "total_statements": format_number_str(
                                len(filtered_rndts_data_df), 0
                            ),  # Total number of bordereaux
                            "total_quantity": format_number_str(
                                filtered_rndts_data_df["quantite"].sum(), 2
                            ),  # Total quantity processed
                        },
                    }
                )

    @staticmethod
    def _preprocess_and_filter_bs_list(
        siret: str,
        dfs_to_process: list[tuple[str, pd.DataFrame]],
        processing_codes: list[str],
        data_date_interval: tuple[datetime, datetime],
        packagings_data_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        bs_dfs = []
        for bs_type, df in dfs_to_process:
            if (df is None) or (len(df) == 0):
                continue

            df_filtered = pd.DataFrame()
            if bs_type != BSFF:
                df_filtered = df[
                    (df["recipient_company_siret"] == siret)
                    & (df["processing_operation_code"].isin(processing_codes))
                    & (df["processed_at"].between(*data_date_interval))
                ].copy()
            else:
                if (packagings_data_df is not None) and (len(packagings_data_df) > 0):
                    df = df.merge(
                        packagings_data_df[
                            [
                                "bsff_id",
                                "acceptation_weight",
                                "operation_date",
                                "operation_code",
                            ]
                        ],
                        left_on="id",
                        right_on="bsff_id",
                        validate="one_to_many",
                    )
                    df = df[
                        (df["recipient_company_siret"] == siret)
                        & (df["operation_code"].isin(processing_codes))
                        & (df["operation_date"].between(*data_date_interval))
                    ]
                    df_filtered = df.groupby("id", as_index=False).agg(
                        {
                            "processing_operation_code": pd.NamedAgg(column="operation_code", aggfunc="max"),
                            "processed_at": pd.NamedAgg(column="operation_date", aggfunc="max"),
                            "quantity_received": pd.NamedAgg(column="acceptation_weight", aggfunc="sum"),
                        }  # type: ignore
                    )  # type: ignore

            if (
                len(df_filtered) > 0
            ):  # If condition is met, it means we found bordereaux without the company having the "rubrique"
                df_filtered["bs_type"] = bs_type.upper()
                bs_dfs.append(df_filtered)

        concat_df = pd.DataFrame()
        if len(bs_dfs) > 0:
            concat_df = pd.concat(bs_dfs).sort_values(["bs_type", "processed_at"])

        return concat_df

    def _preprocess_data(self) -> None:
        self._preprocess_data_multi_rubriques()
        self._preprocess_data_single_rubrique()
        self._preprocess_non_dangerous_rubriques()

    def _check_data_empty(self) -> bool:
        if all(len(e) == 0 for e in self.preprocessed_data.values()):
            return True

        return False

    def _add_stats(self) -> dict[str, list]:
        stats = {"dangerous": [], "non_dangerous": []}

        for item in self.preprocessed_data["dangerous"]:
            if not item:
                continue
            df = item["bs_list"]
            if df is not None:
                rows = []

                if "quantity_refused" in df.columns:
                    df["quantity_received"] = df["quantity_received"] - df["quantity_refused"].fillna(0)

                for e in df.itertuples():
                    row = {
                        "id": e.readable_id if e.bs_type == "BSDD" else e.id,
                        "bs_type": e.bs_type,
                        "waste_code": e.waste_code,
                        "waste_name": e.waste_name
                        if (e.bs_type not in ("BSVHU", "BSDASRI") and not pd.isna(e.waste_name))
                        else None,
                        "operation_code": e.processing_operation_code,
                        "quantity": format_number_str(e.quantity_received, 3)
                        if not pd.isna(e.quantity_received)
                        else None,
                        "processed_at": e.processed_at.strftime("%d/%m/%Y %H:%M")
                        if not pd.isna(e.received_at)
                        else None,
                    }
                    rows.append(row)
                stats["dangerous"].append({**item, "bs_list": rows})

        for item in self.preprocessed_data["non_dangerous"]:
            df = item["statements_list"]
            rows = []
            for e in df.itertuples():
                row = {
                    "waste_code": e.code_dechet,
                    "waste_name": e.denomination_usuelle,
                    "operation_code": e.code_traitement,
                    "quantity": format_number_str(e.quantite, 3) if not pd.isna(e.quantite) else None,
                    "received_at": e.date_reception.strftime("%d/%m/%Y") if not pd.isna(e.date_reception) else None,
                }
                rows.append(row)
            stats["non_dangerous"].append({**item, "statements_list": rows})
        return stats

    def build(self) -> dict:
        self._preprocess_data()

        if not self._check_data_empty():
            return self._add_stats()
        return {}


class LinkedCompaniesProcessor:
    """Component that displays list of ICPE authorized items.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    linked_companies_data: DataFrame
        DataFrame containing list of linked companies
    """

    def __init__(
        self,
        company_siret: str,
        linked_companies_data: pd.DataFrame | None,
    ) -> None:
        self.company_siret = company_siret
        self.linked_companies_data = linked_companies_data

        self.preprocessed_df = None

    def _preprocess_data(self):
        df = self.linked_companies_data
        if df is None:
            return

        df = df[df.siret != self.company_siret]
        if len(df) == 0:
            return

        df = df.sort_values("created_at")

        self.preprocessed_df = df

    def build_context(self):
        data = self.preprocessed_df

        data["created_at"] = data["created_at"].dt.strftime("%d/%m/%Y")

        json_data = {
            "siren": self.company_siret[:9],
            "siret_list": json.loads(data.to_json(orient="records")),
        }
        return json_data

    def _check_empty_data(self) -> bool:
        if self.preprocessed_df is None or len(self.preprocessed_df) == 0:
            return True

        return False

    def build(self):
        self._preprocess_data()

        data = {}
        if not self._check_empty_data():
            data = self.build_context()

        return data


class BsdaWorkerStatsProcessor:
    """Component that compute stats related to worker companies.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bsda_data_df: DataFrame
        DataFrame containing BSDA data.
    data_date_interval: tuple
        Date interval to filter data.
    """

    def __init__(
        self,
        company_siret: str,
        bsda_data_df: pd.DataFrame,
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.bsda_data_df = bsda_data_df
        self.data_date_interval = data_date_interval
        self.company_siret = company_siret

        self.bsda_worker_stats = {
            "signed_producer": None,
            "signed_worker": None,  # and producer
            "signed_transporter": None,  # and worker + producer
            "received": None,
            "processed": None,
            "signed_vs_processed_ratio": None,
            "avg_processing_time_from_emission": None,  # Between signature of producer data and processing date
            "max_processing_time_from_emission": None,  # Between signature of producer data and processing date
            "max_processing_time_from_sending": None,
            "avg_processing_time_from_sending": None,
        }

    def _preprocess_data(self) -> None:
        siret = self.company_siret

        df = self.bsda_data_df
        if len(df) == 0:
            return

        if len(self.bsda_data_df) == 0:
            return

        df = df[df.worker_company_siret == siret]

        if len(df) == 0:
            return

        self.bsda_worker_stats["signed_producer"] = len(
            df[df["emitter_emission_signature_date"].between(*self.data_date_interval)]
        )
        self.bsda_worker_stats["signed_worker"] = len(
            df[
                df["emitter_emission_signature_date"].between(*self.data_date_interval)
                & df["worker_work_signature_date"].between(*self.data_date_interval)
            ]
        )
        self.bsda_worker_stats["signed_transporter"] = len(
            df[
                df["emitter_emission_signature_date"].between(*self.data_date_interval)
                & df["worker_work_signature_date"].between(*self.data_date_interval)
                & df["sent_at"].between(*self.data_date_interval)
            ]
        )
        self.bsda_worker_stats["received"] = len(df[df["received_at"].between(*self.data_date_interval)])
        self.bsda_worker_stats["processed"] = len(df[df["processed_at"].between(*self.data_date_interval)])

        if self.bsda_worker_stats["signed_worker"] > 0:
            self.bsda_worker_stats["signed_vs_processed_ratio"] = format_number_str(
                100 * self.bsda_worker_stats["processed"] / self.bsda_worker_stats["signed_worker"],
                2,
            )

        df_filtered = df[
            df["processed_at"].between(*self.data_date_interval)
            & df["emitter_emission_signature_date"].between(*self.data_date_interval)
            & df["worker_work_signature_date"].between(*self.data_date_interval)
        ]
        times_to_process_from_emission = df_filtered["processed_at"] - df_filtered["emitter_emission_signature_date"]
        max_time_to_process_from_emission = times_to_process_from_emission.max()
        avg_time_to_process_from_emission = times_to_process_from_emission.mean()

        if not pd.isna(max_time_to_process_from_emission):
            self.bsda_worker_stats["max_processing_time_from_emission"] = format_number_str(
                max_time_to_process_from_emission.value / (1e9 * 3600 * 24), 2
            )

        if not pd.isna(avg_time_to_process_from_emission):
            self.bsda_worker_stats["avg_processing_time_from_emission"] = format_number_str(
                avg_time_to_process_from_emission.value / (1e9 * 3600 * 24)
            )

        df_filtered = df[
            df["processed_at"].between(*self.data_date_interval)
            & df["sent_at"].between(*self.data_date_interval)
            & df["worker_work_signature_date"].between(*self.data_date_interval)
        ]
        times_to_process_from_sending = df_filtered["processed_at"] - df_filtered["sent_at"]
        max_time_to_process_from_sending = times_to_process_from_sending.max()
        avg_time_to_process_from_sending = times_to_process_from_sending.mean()

        if not pd.isna(max_time_to_process_from_sending):
            self.bsda_worker_stats["max_processing_time_from_sending"] = format_number_str(
                max_time_to_process_from_sending.value / (1e9 * 3600 * 24), 2
            )

        if not pd.isna(avg_time_to_process_from_sending):
            self.bsda_worker_stats["avg_processing_time_from_sending"] = format_number_str(
                avg_time_to_process_from_sending.value / (1e9 * 3600 * 24)
            )

    def _check_empty_data(self) -> bool:
        if all(e in [None, 0, "0"] for e in self.bsda_worker_stats.values()):
            return True

        return False

    def build(self):
        self._preprocess_data()

        res = {}

        if not self._check_empty_data():
            res = self.bsda_worker_stats
        return res


class TransporterBordereauxStatsProcessor:
    """Component that compute statistics about number of bordereaux as transporter company and corresponding quantities.

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

        self.transported_bordereaux_stats = {
            BSDD: {},
            BSDD_NON_DANGEROUS: {},
            BSDA: {},
            BSFF: {},
            BSDASRI: {},
            BSVHU: {},
        }

    def _preprocess_bs_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it to be displayed."""
        transporter_data_dfs = self.transporters_data_df
        bs_data_dfs = self.bs_data_dfs

        for bs_type, df in chain(transporter_data_dfs.items(), bs_data_dfs.items()):
            df = df[
                df["sent_at"].between(*self.data_date_interval)
                & (df["transporter_company_siret"] == self.company_siret)
            ]

            if len(df) > 0:
                quantity_col = "quantity_received"
                id_col = "bs_id" if bs_type in [BSDD, BSDD_NON_DANGEROUS, BSDA, BSFF] else "id"

                num_bordereaux = df[id_col].nunique()
                quantity = df[quantity_col].sum()
                self.transported_bordereaux_stats[bs_type]["count"] = num_bordereaux
                self.transported_bordereaux_stats[bs_type]["quantity"] = format_number_str(quantity, 2)

    def _check_data_empty(self) -> bool:
        if all((e is None) or (e == {}) for e in self.transported_bordereaux_stats.values()):
            return True

        return False

    def build(self):
        self._preprocess_bs_data()

        data = {}
        if not self._check_data_empty():
            data = self.transported_bordereaux_stats

        return data


class FollowedWithPNTTDTableProcessor:
    """Component that displays an exhaustive tables of BSDD followed by PNTTD.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
        Only BSDD and BSDD non dangerous.
    data_date_interval: tuple
        Date interval to filter data.
    waste_codes_df: DataFrame
        DataFrame containing list of waste codes with their descriptions. It is the waste nomenclature.
    """

    def __init__(
        self,
        company_siret: str,
        bs_data_dfs: Dict[str, pd.DataFrame],
        data_date_interval: tuple[datetime, datetime],
        waste_codes_df: pd.DataFrame,
    ) -> None:
        self.bs_data_dfs = bs_data_dfs
        self.data_date_interval = data_date_interval
        self.waste_codes_df = waste_codes_df
        self.company_siret = company_siret

        self.preprocessed_df = None

    def _preprocess_data(self) -> None:
        siret = self.company_siret

        dfs_to_concat = [df for df in self.bs_data_dfs.values() if df is not None]

        if len(dfs_to_concat) == 0:
            self.preprocessed_df = pd.DataFrame()
            return

        df = pd.concat(dfs_to_concat)

        df = df[
            (df["recipient_company_siret"] == siret)
            & (df["status"] == "FOLLOWED_WITH_PNTTD")
            & df["processed_at"].between(*self.data_date_interval)
        ]

        if len(df) > 0:
            df["foreign_org_id"] = df.apply(
                lambda x: x["next_destination_company_siret"]
                if not (pd.isna(x["next_destination_company_siret"]) or x["next_destination_company_siret"] == "")
                else x["next_destination_company_vat_number"],
                axis=1,
            )

            # Handle quantity refused
            df["quantity_received"] = df["quantity_received"] - df["quantity_refused"].fillna(0)

            # We compute the quantity by waste codes
            df_grouped = df.groupby(
                [
                    "foreign_org_id",
                    "waste_code",
                    "next_destination_processing_operation",
                ],
                as_index=False,
            ).agg(
                quantity=pd.NamedAgg("quantity_received", "sum"),
                destination_country=pd.NamedAgg("next_destination_company_country", "max"),
            )
            # We add the waste code description from the waste nomenclature
            final_df = pd.merge(
                df_grouped,
                self.waste_codes_df,
                left_on="waste_code",
                right_index=True,
                how="left",
                validate="many_to_one",
            )

            company_names = (
                df.groupby(by="foreign_org_id")["next_destination_company_name"].max().rename("destination_name")
            )

            final_df = final_df.merge(company_names, left_on="foreign_org_id", right_index=True)

            final_df["quantity"] = final_df["quantity"].apply(lambda x: format_number_str(x, 2))
            final_df["description"] = final_df["description"].fillna("")
            final_df = final_df.rename_axis(None, axis=0)
            self.preprocessed_df = final_df[
                [
                    "foreign_org_id",
                    "destination_name",
                    "destination_country",
                    "waste_code",
                    "description",
                    "next_destination_processing_operation",
                    "quantity",
                ]
            ].sort_values(by=["foreign_org_id", "waste_code"])

    def _check_empty_data(self) -> bool:
        if self.preprocessed_df is None:
            return True

        return False

    def build_context(self):
        return self.preprocessed_df.to_dict("records")

    def build(self):
        self._preprocess_data()

        res = {}

        if not self._check_empty_data():
            res = self.build_context()
        return res


class GistridStatsProcessor:
    """Component that compute statistics about Gistrid/PNTTD data.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    gistrid_data_df: pd.DataFrame
        DataFrame containing Gistrid notifications.
    """

    def __init__(self, company_siret: str, gistrid_data_df: pd.DataFrame | None) -> None:
        self.company_siret = company_siret
        self.gistrid_data_df = gistrid_data_df

        self.gistrid_stats = {}

    def _preprocess_gistrid_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it to be displayed."""
        df = self.gistrid_data_df
        if (df is None) or (len(df) == 0):
            return

        df = self.gistrid_data_df
        df["annee_fin_autorisation"] = df["date_autorisee_fin_transferts"].str[-2:]

        import_data = df[df["siret_installation_traitement"] == self.company_siret]

        def parse_codes(x):
            codes = set()
            for codes_str in x:
                codes.update(codes_str.split(", "))
            return ", ".join(sorted(codes))

        import_data_grouped = (
            import_data.groupby(["annee_fin_autorisation", "numero_gistrid_notifiant"], as_index=False)
            .aggregate(
                nom_origine=pd.NamedAgg(column="nom_notifiant", aggfunc="max"),
                pays_origine=pd.NamedAgg(column="pays_notifiant", aggfunc="max"),
                quantites_recues=pd.NamedAgg(column="somme_quantites_recues", aggfunc="sum"),
                nombre_transferts=pd.NamedAgg(column="nombre_transferts_receptionnes", aggfunc="sum"),
                codes_dechets=pd.NamedAgg(column="code_ced", aggfunc=parse_codes),
                codes_operations=pd.NamedAgg(column="code_d_r", aggfunc=parse_codes),
            )
            .sort_values("annee_fin_autorisation")
        )
        import_data_grouped["quantites_recues"] = import_data_grouped["quantites_recues"].apply(format_number_str, 2)
        if len(import_data_grouped) > 0:
            self.gistrid_stats["import"] = import_data_grouped.to_dict(orient="records")
            self.gistrid_stats["numero_gistrid"] = import_data["numero_gistrid_installation_traitement"].iloc[0]

        export_data = df[df["siret_notifiant"] == self.company_siret]

        export_data_grouped = (
            export_data.groupby(
                ["annee_fin_autorisation", "numero_gistrid_installation_traitement"],
                as_index=False,
            )
            .aggregate(
                nom_destination=pd.NamedAgg(column="nom_installation_traitement", aggfunc="max"),
                pays_destination=pd.NamedAgg(column="pays_installation_traitement", aggfunc="max"),
                quantites_recues=pd.NamedAgg(column="somme_quantites_recues", aggfunc="sum"),
                nombre_transferts=pd.NamedAgg(column="nombre_transferts_receptionnes", aggfunc="sum"),
                codes_dechets=pd.NamedAgg(column="code_ced", aggfunc=parse_codes),
                codes_operations=pd.NamedAgg(column="code_d_r", aggfunc=parse_codes),
            )
            .sort_values("annee_fin_autorisation")
        )
        export_data_grouped["quantites_recues"] = export_data_grouped["quantites_recues"].apply(format_number_str, 2)
        if len(export_data_grouped) > 0:
            self.gistrid_stats["export"] = export_data_grouped.to_dict(orient="records")
            self.gistrid_stats["numero_gistrid"] = export_data["numero_gistrid_notifiant"].iloc[0]

    def _check_data_empty(self) -> bool:
        if len(self.gistrid_stats) == 0:
            return True

        return False

    def build(self):
        self._preprocess_gistrid_data()

        data = {}
        if not self._check_data_empty():
            data = self.gistrid_stats

        return data


class RNDTSStatsProcessor:
    """Component that displays aggregated data about RNDTS non dangerous waste data.

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
        rndts_incoming_data: pd.DataFrame,
        rndts_outgoing_data: pd.DataFrame,
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.rndts_incoming_data = rndts_incoming_data
        self.rndts_outgoing_data = rndts_outgoing_data
        self.data_date_interval = data_date_interval

        # Init all statistics
        self.stats = {
            "total_weight_incoming": 0,
            "total_weight_outgoing": 0,
            "bar_size_weight_incoming": None,
            "bar_size_weight_outgoing": None,
            "has_weight": None,
            "total_volume_incoming": 0,
            "total_volume_outgoing": 0,
            "bar_size_volume_incoming": None,
            "bar_size_volume_outgoing": None,
            "has_volume": None,
            "total_statements_incoming": 0,
            "total_statements_outgoing": 0,
        }

    def _check_data_empty(self) -> bool:
        # If all values after preprocessing are empty, then output data will be empty
        if all((e == 0) or (e is None) for e in self.stats.values()):
            return True

        return False

    def _preprocess_data(self) -> None:
        incoming_data = self.rndts_incoming_data
        outgoing_data = self.rndts_outgoing_data

        if incoming_data is not None:
            incoming_data = incoming_data[
                incoming_data["date_reception"].between(*self.data_date_interval)
                & (incoming_data["etablissement_numero_identification"] == self.company_siret)
            ]
            if len(incoming_data) > 0:
                self.stats["total_statements_incoming"] = incoming_data["id"].nunique()

                for unit, key in [("T", "weight"), ("M3", "volume")]:
                    total = incoming_data[incoming_data["unite"] == unit]["quantite"].sum()
                    if total is not None:
                        self.stats[f"total_{key}_incoming"] = total

        if outgoing_data is not None:
            colname = "producteur_numero_identification"
            if "producteur_numero_identification" not in outgoing_data.columns:
                colname = "etablissement_numero_identification"  # SSD case
            outgoing_data = outgoing_data[
                outgoing_data["date_expedition"].between(*self.data_date_interval)
                & (outgoing_data[colname] == self.company_siret)
            ]
            if len(outgoing_data) > 0:
                self.stats["total_statements_outgoing"] = outgoing_data["id"].nunique()

                for unit, key in [("T", "weight"), ("M3", "volume")]:
                    total = outgoing_data[outgoing_data["unite"] == unit]["quantite"].sum()
                    if total is not None:
                        self.stats[f"total_{key}_outgoing"] = total

        for key in ["weight", "volume"]:
            incoming_bar_size = 0
            outgoing_bar_size = 0

            total_quantity_incoming = self.stats[f"total_{key}_incoming"]
            total_quantity_outgoing = self.stats[f"total_{key}_outgoing"]
            if not (total_quantity_incoming == total_quantity_outgoing == 0):
                # The bar sizes are relative to the largest quantity.
                # Size is expressed as percentage of the component width.
                if total_quantity_incoming > total_quantity_outgoing:
                    incoming_bar_size = 100
                    outgoing_bar_size = int(100 * (total_quantity_outgoing / total_quantity_incoming))
                else:
                    incoming_bar_size = int(100 * (total_quantity_incoming / total_quantity_outgoing))
                    outgoing_bar_size = 100
                self.stats[f"has_{key}"] = True
            else:
                self.stats[f"has_{key}"] = False
            self.stats[f"bar_size_{key}_incoming"] = incoming_bar_size
            self.stats[f"bar_size_{key}_outgoing"] = outgoing_bar_size

    def build_context(self):
        # We use the format_number_str only on variables that holds
        # quantity values.

        precisions = {
            "total_weight_incoming": 2,
            "total_weight_outgoing": 2,
            "bar_size_weight_incoming": 2,
            "bar_size_weight_outgoing": 2,
            "total_volume_incoming": 2,
            "total_volume_outgoing": 2,
            "bar_size_volume_incoming": 2,
            "bar_size_volume_outgoing": 2,
            "total_statements_incoming": 0,
            "total_statements_outgoing": 0,
        }

        ctx = {
            k: format_number_str(v, precisions[k])
            if (isinstance(v, numbers.Number) and not isinstance(v, bool))
            else v
            for k, v in self.stats.items()
        }

        return ctx

    def build(self):
        self._preprocess_data()

        data = {}
        if not self._check_data_empty():
            data = self.build_context()
        return data


class IntermediaryBordereauxStatsProcessor:
    """Component that compute statistics about number of bordereaux as "eco-organisme" and corresponding quantities.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    transporters_data_df: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau transported data.
        Correspond to the new way of managing transporters in Trackdéchets.
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
            BSDD: {},
            BSDD_NON_DANGEROUS: {},
            BSDA: {},
            BSDASRI: {},
        }

    def _preprocess_bs_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it to be displayed."""
        bs_data_dfs = self.bs_data_dfs.copy()

        for bs_type, df in bs_data_dfs.items():
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
                num_bordereaux = df["id"].nunique()

                # handle quantity refused
                if bs_type in [BSDD, BSDD_NON_DANGEROUS]:
                    df["quantity_received"] = df["quantity_received"] - df["quantity_refused"].fillna(0)

                quantity = df.drop_duplicates("id")["quantity_received"].sum()
                self.bordereaux_stats[bs_type]["count"] = format_number_str(num_bordereaux, 0)
                self.bordereaux_stats[bs_type]["quantity"] = format_number_str(quantity, 2)

    def _check_data_empty(self) -> bool:
        if all((e is None) or (e == {}) for e in self.bordereaux_stats.values()):
            return True

        return False

    def build(self):
        self._preprocess_bs_data()

        data = {}
        if not self._check_data_empty():
            data = self.bordereaux_stats

        return data


class IncineratorOutgoingWasteProcessor:
    """Component that aggregate data to show a table of outgoing dangerous and non dangerous waste for incinerators.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    bs_data_dfs: dict
        Dict with key being the 'bordereau' type and values the DataFrame containing the bordereau data.
    icpe_data: DataFrame
        DataFrame containing list of ICPE authorized items
    rndts_outgoing_data: DataFrame
        DataFrame containing data for outgoing non dangerous waste (from RNDTS).
    data_date_interval: tuple
        Date interval to filter data.
    """

    def __init__(
        self,
        company_siret: str,
        bs_data_dfs: Dict[str, pd.DataFrame],
        transporters_data_df: Dict[str, pd.DataFrame],  # Handling new multi-modal Trackdéchets feature
        icpe_data: pd.DataFrame | None,
        rndts_outgoing_data: pd.DataFrame | None,
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.bs_data_dfs = bs_data_dfs
        self.transporters_data_df = transporters_data_df
        self.icpe_data = icpe_data
        self.rndts_outgoing_data = rndts_outgoing_data
        self.data_date_interval = data_date_interval

        self.preprocessed_data = {"dangerous": pd.DataFrame(), "non_dangerous": pd.DataFrame()}

    def _preprocess_bs_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it to be displayed."""
        bs_data_dfs = self.bs_data_dfs

        dfs_to_concat = []
        for bs_type, df in bs_data_dfs.items():
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
                df["sent_at"].between(*self.data_date_interval) & (df["emitter_company_siret"] == self.company_siret)
            ]
            df = df.drop_duplicates("id")

            if len(df) > 0:
                dfs_to_concat.append(df)

        if len(dfs_to_concat) > 0:
            concat_df = pd.concat(dfs_to_concat)
            concat_df["waste_name"] = concat_df["waste_name"].fillna("")

            # Handle quantity refused
            if "quantity_refused" in concat_df.columns:
                concat_df["quantity_received"] = concat_df["quantity_received"] - concat_df["quantity_refused"].fillna(
                    0
                )

            aggregated_data_df = (
                concat_df.groupby(
                    ["waste_code", "recipient_company_siret", "processing_operation_code"], as_index=False
                )
                .aggregate(
                    quantity=pd.NamedAgg(column="quantity_received", aggfunc="sum"),
                    waste_name=pd.NamedAgg(column="waste_name", aggfunc="max"),
                )
                .sort_values(by=["waste_code", "recipient_company_siret", "quantity"], ascending=[True, True, False])
            )

            if len(aggregated_data_df) > 0:
                self.preprocessed_data["dangerous"] = aggregated_data_df

    def _preprocess_rndts_statements_data(self) -> None:
        """Preprocess raw RNDTS statements data to prepare it to be displayed."""
        rndts_data = self.rndts_outgoing_data

        if (rndts_data is None) or (len(rndts_data) == 0):
            return

        rndts_data["denomination_usuelle"] = rndts_data["denomination_usuelle"].fillna("")
        aggregated_data_df = (
            rndts_data[
                (rndts_data["producteur_numero_identification"] == self.company_siret)
                & (rndts_data["date_expedition"].between(*self.data_date_interval))
            ]
            .groupby(["code_dechet", "destinataire_numero_identification", "code_traitement", "unite"], as_index=False)
            .agg(
                quantite=pd.NamedAgg(column="quantite", aggfunc="sum"),
                denomination_usuelle=pd.NamedAgg(column="denomination_usuelle", aggfunc="max"),
            )
            .sort_values(
                by=["code_dechet", "destinataire_numero_identification", "quantite"], ascending=[True, True, False]
            )
        )

        if len(aggregated_data_df) > 0:
            self.preprocessed_data["non_dangerous"] = aggregated_data_df

    def is_incinerator(self, dangerous_waste: bool) -> bool:
        rubrique = "2770" if dangerous_waste else "2771"
        icpe_data = self.icpe_data
        if (icpe_data is None) or (len(icpe_data) == 0):
            return False

        return (icpe_data["rubrique"] == rubrique).any()

    def _preprocess_data(self):
        if self.is_incinerator(dangerous_waste=True):
            self._preprocess_bs_data()
        if self.is_incinerator(dangerous_waste=False):
            self._preprocess_rndts_statements_data()

    def _check_data_empty(self) -> bool:
        if all((e is None) or (len(e) == 0) for e in self.preprocessed_data.values()):
            return True

        return False

    def _serialize_stats(self) -> dict:
        res = {"dangerous": [], "non_dangerous": []}

        for _, row in self.preprocessed_data["dangerous"].iterrows():
            res["dangerous"].append(
                {
                    "waste_code": row.waste_code,
                    "waste_name": row.waste_name,
                    "destination_company_siret": row.recipient_company_siret,
                    "processing_opration": row.processing_operation_code,
                    "quantity": format_number_str(row.quantity, 2),
                }
            )

        for _, row in self.preprocessed_data["non_dangerous"].iterrows():
            res["non_dangerous"].append(
                {
                    "waste_code": row.code_dechet,
                    "waste_name": row.denomination_usuelle,
                    "destination_company_siret": row.destinataire_numero_identification,
                    "unit": row.unite,
                    "processing_opration": row.code_traitement,
                    "quantity": format_number_str(row.quantite, 2),
                }
            )
        return res

    def build(self):
        self._preprocess_data()

        data = {}
        if not self._check_data_empty():
            data = self._serialize_stats()

        return data


class SSDProcessor:
    """Component that aggregate data to show a table of SSD quantities by waste code.

    Parameters
    ----------
    company_siret: str
        SIRET number of the establishment for which the data is displayed (used for data preprocessing).
    ssd_data: DataFrame
        DataFrame containing list of ssd statements.
    data_date_interval: tuple
        Date interval to filter data.
    """

    def __init__(
        self,
        company_siret: str,
        ssd_data: pd.DataFrame | None,
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.ssd_data = ssd_data
        self.data_date_interval = data_date_interval

        self.preprocessed_data = pd.DataFrame()

    def _preprocess_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it to be displayed."""
        ssd_data_df = self.ssd_data

        if ssd_data_df is None:
            return

        ssd_data = ssd_data_df[
            (ssd_data_df["etablissement_numero_identification"] == self.company_siret)
            & (ssd_data_df["date_expedition"].between(*self.data_date_interval))
        ]

        if len(ssd_data) > 0:
            ssd_data_agg = ssd_data.groupby(["code_dechet", "nature", "unite"], as_index=False).agg(
                quantite=pd.NamedAgg(column="quantite", aggfunc="sum"),
                denomination_usuelle=pd.NamedAgg(column="denomination_usuelle", aggfunc="max"),
            )
            ssd_data_agg["unite"] = ssd_data_agg["unite"].str.lower()
            ssd_data_agg = ssd_data_agg.sort_values(["code_dechet", "unite"])

            self.preprocessed_data = ssd_data_agg

    def _check_data_empty(self) -> bool:
        if len(self.preprocessed_data) == 0:
            return True

        return False

    def _serialize_stats(self) -> list[dict]:
        res = []

        for _, row in self.preprocessed_data.iterrows():
            res.append(
                {
                    "waste_code": row.code_dechet,
                    "waste_name": row.denomination_usuelle,
                    "nature": row.nature,
                    "quantity": format_number_str(row.quantite, 2),
                    "unit": row.unite,
                }
            )

        return res

    def build(self):
        self._preprocess_data()

        data = {}
        if not self._check_data_empty():
            data = self._serialize_stats()

        return data


class RNDTSTransporterStatsProcessor:
    """Component that compute statistics about number of RDNTS statements as transporter company and corresponding quantities.

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
        rndts_data: Dict[str, pd.DataFrame | None],
        data_date_interval: tuple[datetime, datetime],
    ) -> None:
        self.company_siret = company_siret
        self.rndts_data = rndts_data
        self.data_date_interval = data_date_interval

        self.transported_statements_stats = {
            "ndw_incoming": {},
            "ndw_outgoing": {},
            "excavated_land_incoming": {},
            "excavated_land_outgoing": {},
        }

    def _preprocess_bs_data(self) -> None:
        """Preprocess raw 'bordereaux' data to prepare it to be displayed."""

        rndts_data = self.rndts_data

        for key, date_col in [
            ("ndw_incoming", "date_reception"),
            ("ndw_outgoing", "date_expedition"),
            ("excavated_land_incoming", "date_reception"),
            ("excavated_land_outgoing", "date_expedition"),
        ]:
            df = rndts_data[key]
            if (df is None) or (len(df) == 0):
                continue

            df = df[
                df[date_col].between(*self.data_date_interval)
                & (df["numeros_indentification_transporteurs"].apply(lambda x: self.company_siret in x))
            ]

            if len(df) > 0:
                num_statements = df["id"].nunique()
                mass_quantity = df[df["unite"] == "T"]["quantite"].sum()
                volume_quantity = df[df["unite"] == "M3"]["quantite"].sum()
                self.transported_statements_stats[key]["count"] = num_statements
                self.transported_statements_stats[key]["mass_quantity"] = format_number_str(mass_quantity, 2)
                self.transported_statements_stats[key]["volume_quantity"] = format_number_str(volume_quantity, 2)

    def _check_data_empty(self) -> bool:
        if all((e is None) or (e == {}) for e in self.transported_statements_stats.values()):
            return True

        return False

    def build(self):
        self._preprocess_bs_data()

        data = {}
        if not self._check_data_empty():
            data = self.transported_statements_stats

        return data
