from typing import Dict, List, Tuple

import pandas as pd
from django.utils import timezone

from .constants import BS_TYPES_WITH_MULTIMODAL_TRANSPORT, BSDA, BSDASRI, BSDD, BSDD_NON_DANGEROUS, BSFF, BSVHU
from .data_extract import (
    load_and_preprocess_regions_geographical_data,
    load_departements_regions_data,
    load_mapping_rubrique_processing_operation_code,
    load_waste_code_data,
)
from .database import (
    build_bsda_query,
    build_bsda_transporter_query,
    build_bsdasri_query,
    build_bsdd_non_dangerous_query,
    build_bsdd_non_dangerous_transporter_query,
    build_bsdd_query,
    build_bsdd_transporter_query,
    build_bsff_packagings_query,
    build_bsff_query,
    build_bsff_transporter_query,
    build_bsvhu_query,
    build_query_company,
    build_revised_bsda_query,
    build_revised_bsdasri_query,
    build_revised_bsdd_query,
    get_agreement_data,
    get_gistrid_data,
    get_icpe_data,
    get_icpe_item_data,
    get_linked_companies_data,
    get_rndts_excavated_land_data,
    get_rndts_ndw_data,
    get_ssd_data,
)
from .graph_processors.html_components_processors import (
    BsdaWorkerStatsProcessor,
    BsdCanceledTableProcessor,
    BsdStatsProcessor,
    FollowedWithPNTTDTableProcessor,
    GistridStatsProcessor,
    ICPEItemsProcessor,
    IncineratorOutgoingWasteProcessor,
    IntermediaryBordereauxStatsProcessor,
    LinkedCompaniesProcessor,
    PrivateIndividualsCollectionsTableProcessor,
    QuantityOutliersTableProcessor,
    ReceiptAgrementsProcessor,
    RNDTSStatsProcessor,
    RNDTSTransporterStatsProcessor,
    SameEmitterRecipientTableProcessor,
    SSDProcessor,
    StorageStatsProcessor,
    TraceabilityInterruptionsProcessor,
    TransporterBordereauxStatsProcessor,
    WasteFlowsTableProcessor,
    WasteIsDangerousStatementsProcessor,
    WasteProcessingWithoutICPERubriqueProcessor,
)
from .graph_processors.plotly_components_processors import (
    BsdaWorkerQuantityProcessor,
    BsdQuantitiesGraph,
    BsdTrackedAndRevisedProcessor,
    ICPEAnnualItemProcessor,
    ICPEDailyItemProcessor,
    IntermediaryBordereauxCountsGraphProcessor,
    IntermediaryBordereauxQuantitiesGraphProcessor,
    RNDTSQuantitiesGraphProcessor,
    RNDTSStatementsGraphProcessor,
    RNDTSTransporterQuantitiesGraphProcessor,
    RNDTSTransporterStatementsStatsGraphProcessor,
    TransportedQuantitiesGraphProcessor,
    TransporterBordereauxGraphProcessor,
    WasteOriginProcessor,
    WasteOriginsMapProcessor,
)
from .models import ComputedInspectionData
from .utils import (
    get_quantity_variable_names,
    to_verbose_collector_types,
    to_verbose_company_types,
    to_verbose_waste_processor_types,
)

WASTE_CODES_DATA = load_waste_code_data()
DEPARTEMENTS_REGION_DATA = load_departements_regions_data()
REGIONS_GEODATA = load_and_preprocess_regions_geographical_data()
PROCESSING_OPERATION_CODE_RUBRIQUE_MAPPING = load_mapping_rubrique_processing_operation_code()


def get_outliers_datetimes_df(
    df: pd.DataFrame, date_columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """For a given DataFrame, separate lines with date outliers
    from lines with consistent (parsable) dates in provided list of date columns.

    Parameters
    ----------
    df : DataFrame
        DataFrame with raw (str) date data.
    date_columns : list of str
        Names of columns to parse as dates in pandas (time-zone aware dates are casted to UTC).

    Returns
    -------
    Tuple consisting of :
    1. DataFrame with consistent dates.
    2. Dict with keys being date column names and values being the DataFrame's lines with inconsistent date for this column.
    """
    df = df.copy()
    outliers = {}
    idx_with_outliers = set()

    for colname in date_columns:
        time_data = pd.to_datetime(df[colname], errors="coerce")
        outliers_df = df[time_data.isna() & (~df[colname].isin(["None", "NaT"]))]
        if len(outliers_df) > 0:
            idx_with_outliers.update(outliers_df.index.tolist())
            outliers[colname] = outliers_df

    if len(idx_with_outliers) > 0:
        df = df[~df.index.isin(idx_with_outliers)]

    for colname in date_columns:
        df[colname] = pd.to_datetime(df[colname].replace(["None", "NaT"], pd.NaT), utc=True)

    df["created_at"] = pd.to_datetime(df["created_at"].replace(["None", "NaT"], pd.NaT), utc=True)
    return df, outliers


bsds_config = [
    {
        "bsd_type": BSDD,
        "bs_data": build_bsdd_query,
        "bs_revised_data": build_revised_bsdd_query,
        "bs_transporter_data": build_bsdd_transporter_query,
    },
    {
        "bsd_type": BSDD_NON_DANGEROUS,
        "bs_data": build_bsdd_non_dangerous_query,
        "bs_revised_data": build_revised_bsdd_query,
        "bs_transporter_data": build_bsdd_non_dangerous_transporter_query,
    },
    {
        "bsd_type": BSDA,
        "bs_data": build_bsda_query,
        "bs_revised_data": build_revised_bsda_query,
        "bs_transporter_data": build_bsda_transporter_query,
    },
    {
        "bsd_type": BSDASRI,
        "bs_data": build_bsdasri_query,
        "bs_revised_data": build_revised_bsdasri_query,
    },
    {
        "bsd_type": BSFF,
        "bs_data": build_bsff_query,
        "packagings_data": build_bsff_packagings_query,
        "bs_transporter_data": build_bsff_transporter_query,
    },
    {"bsd_type": BSVHU, "bs_data": build_bsvhu_query},
]


class SheetProcessor:
    def __init__(self, computed_pk, force_recompute=False):
        self.computed = ComputedInspectionData.objects.get(pk=computed_pk)
        self.force_recompute = force_recompute

        self.data_start_date = self.computed.data_start_date.replace(tzinfo=None)
        self.data_end_date = self.computed.data_end_date.replace(tzinfo=None)
        self.siret = self.computed.org_id

        self.company_data = pd.DataFrame()

        self.all_bsd_data_empty = True
        self.all_rndts_data_empty = True

        self.company_id = None
        self.receipts_agreements_data = {}
        self.linked_companies_data = None
        self.bs_dfs = {}
        self.revised_bs_dfs = {}
        self.transporter_data_dfs = {}
        self.bsff_packagings_df = None
        self.icpe_data = None
        self.icpe_rubriques_data = {}
        self.rndts_data = {
            "ndw_incoming": None,
            "ndw_outgoing": None,
            "excavated_land_incoming:": None,
            "excavated_land_outgoing": None,
            "ssd_data": None,
        }

    def _extract_data(self):
        self._extract_company_data()
        self._extract_trackdechets_data()
        self._extract_icpe_data()
        self._extract_rndts_data()

    def _extract_company_data(self):
        company_data_df = build_query_company(siret=self.siret, date_params=["created_at"])
        self.company_data = company_data_df
        self.company_id = company_data_df["id"].item()

        self.receipts_agreements_data = get_agreement_data(company_data_df)
        self.linked_companies_data = get_linked_companies_data(self.siret)

    def _extract_trackdechets_data(self):
        for bsd_config in bsds_config:
            bsd_type = bsd_config["bsd_type"]
            # compute and store df in a dict
            df = bsd_config["bs_data"](
                siret=self.computed.org_id,
            )
            self.bs_dfs[bsd_type] = df

            bs_revised_data = bsd_config.get("bs_revised_data", None)
            if bs_revised_data:
                revised_df = bs_revised_data(
                    company_id=self.company_id,
                )
                if len(revised_df) > 0:
                    self.revised_bs_dfs[bsd_type] = revised_df

            bs_transporter_data = bsd_config.get("bs_transporter_data", None)
            if bs_transporter_data is not None:
                transporter_data_df = bs_transporter_data(
                    siret=self.computed.org_id,
                )
                if len(transporter_data_df) > 0:
                    self.transporter_data_dfs[bsd_type] = transporter_data_df

            if bsd_type == BSFF:
                bsff_packagings_data = bsd_config.get("packagings_data")(
                    siret=self.computed.org_id,
                )  # type: ignore
                if len(bsff_packagings_data) > 0:
                    self.bsff_packagings_df = bsff_packagings_data

    def _extract_icpe_data(self):
        self.icpe_data = get_icpe_data(self.computed.org_id)

        for rubrique in ["2770", "2790", "2760-1", "2771", "2791", "2760-2"]:
            self.icpe_rubriques_data[rubrique] = get_icpe_item_data(siret=self.siret, rubrique=rubrique)

    def _extract_rndts_data(self):
        rndts_ndw_incoming_data, rndts_ndw_outgoing_data = get_rndts_ndw_data(self.siret)
        self.rndts_data["ndw_incoming"] = rndts_ndw_incoming_data
        self.rndts_data["ndw_outgoing"] = rndts_ndw_outgoing_data

        rndts_excavated_land_incoming_data, rndts_excavated_land_outgoing_data = get_rndts_excavated_land_data(
            self.siret
        )
        self.rndts_data["excavated_land_incoming"] = rndts_excavated_land_incoming_data
        self.rndts_data["excavated_land_outgoing"] = rndts_excavated_land_outgoing_data
        self.rndts_data["ssd_data"] = get_ssd_data(self.siret)

    def _process_company_data(self):
        company_data_df = self.company_data
        company_values = self.company_data.iloc[0]
        self.company_id = company_data_df.iloc[0].id
        self.computed.company_name = company_values.get("name")
        self.computed.company_address = company_values.get("address")
        self.computed.company_profiles = to_verbose_company_types(company_values.get("company_types"))
        self.computed.company_collector_profiles = to_verbose_collector_types(company_values.get("collector_types"))
        self.computed.company_waste_processor_profiles = to_verbose_waste_processor_types(
            company_values.get("waste_processor_types")
        )
        self.computed.company_created_at = company_values.get("created_at")

        self.computed.save()

    def _build_components(self):
        # Build each plotly graph component
        self._build_graph_components()
        # Build each component that does not contain plotly graph
        self._build_html_components()

    def _build_graph_components(self):
        data_date_interval = (self.data_start_date, self.data_end_date)

        for bs_type, df in self.bs_dfs.items():
            if not len(df):
                continue

            created_rectified_graph = BsdTrackedAndRevisedProcessor(
                self.siret,
                df,
                data_date_interval,
                self.revised_bs_dfs.get(bs_type, None),
            )
            created_rectified_graph_data = created_rectified_graph.build()

            if created_rectified_graph_data:
                self.all_bsd_data_empty = False

            setattr(
                self.computed,
                f"{bs_type}_created_rectified_data",
                created_rectified_graph_data,
            )

            quantity_variables = get_quantity_variable_names(bs_type)

            packaging_data = None
            if bs_type == BSFF:
                packaging_data = self.bsff_packagings_df

            stock_graph = BsdQuantitiesGraph(
                self.siret,
                bs_type,
                df,
                data_date_interval,
                quantity_variables_names=quantity_variables,
                packagings_data=packaging_data,
            )
            stock_graph_data = stock_graph.build()
            if stock_graph_data:
                self.all_bsd_data_empty = False
            setattr(self.computed, f"{bs_type}_stock_data", stock_graph_data)

        transporter_bordereaux_graph = TransporterBordereauxGraphProcessor(
            company_siret=self.siret,
            transporters_data_df=self.transporter_data_dfs,
            bs_data_dfs={k: v for k, v in self.bs_dfs.items() if k not in BS_TYPES_WITH_MULTIMODAL_TRANSPORT},
            data_date_interval=data_date_interval,
        )
        self.computed.transporter_bordereaux_stats_graph_data = transporter_bordereaux_graph.build()

        quantities_transported_graph = TransportedQuantitiesGraphProcessor(
            company_siret=self.siret,
            transporters_data_df=self.transporter_data_dfs,
            bs_data_dfs={k: v for k, v in self.bs_dfs.items() if k not in BS_TYPES_WITH_MULTIMODAL_TRANSPORT},
            data_date_interval=data_date_interval,
            packagings_data_df=self.bsff_packagings_df,
        )
        self.computed.quantities_transported_stats_graph_data = quantities_transported_graph.build()

        waste_origin = WasteOriginProcessor(
            self.siret,
            self.bs_dfs,
            DEPARTEMENTS_REGION_DATA,
            data_date_interval,
        )
        self.computed.waste_origin_data = waste_origin.build()

        waste_origin_map = WasteOriginsMapProcessor(
            self.siret,
            self.bs_dfs,
            DEPARTEMENTS_REGION_DATA,
            REGIONS_GEODATA,
            data_date_interval,
        )
        self.computed.waste_origin_map_data = waste_origin_map.build()

        for rubrique, processor in [
            ("2770", ICPEDailyItemProcessor),
            ("2790", ICPEDailyItemProcessor),
            ("2760-1", ICPEAnnualItemProcessor),
            ("2771", ICPEDailyItemProcessor),
            ("2791", ICPEDailyItemProcessor),
            ("2760-2", ICPEAnnualItemProcessor),
        ]:
            icpe_rubrique_data = self.icpe_rubriques_data[rubrique]
            icpe_rubrique_graph = processor(
                icpe_rubrique_data,
            )
            icpe_rubrique_graph_data = icpe_rubrique_graph.build()
            setattr(self.computed, f"icpe_{rubrique.replace('-', '_')}_data", icpe_rubrique_graph_data)

        non_dangerous_waste_quantities_graph = RNDTSQuantitiesGraphProcessor(
            self.siret, self.rndts_data["ndw_incoming"], self.rndts_data["ndw_outgoing"], data_date_interval
        )
        self.computed.non_dangerous_waste_quantities_graph_data = non_dangerous_waste_quantities_graph.build()
        if self.computed.non_dangerous_waste_quantities_graph_data:
            self.all_rndts_data_empty = False

        non_dangerous_waste_statements_graph = RNDTSStatementsGraphProcessor(
            self.siret,
            self.rndts_data["ndw_incoming"],
            self.rndts_data["ndw_outgoing"],
            "non_dangerous_waste",
            data_date_interval,
        )
        self.computed.non_dangerous_waste_statements_graph_data = non_dangerous_waste_statements_graph.build()
        if self.computed.non_dangerous_waste_statements_graph_data:
            self.all_rndts_data_empty = False

        excavated_land_quantities_graph = RNDTSQuantitiesGraphProcessor(
            self.siret,
            self.rndts_data["excavated_land_incoming"],
            self.rndts_data["excavated_land_outgoing"],
            data_date_interval,
        )
        self.computed.excavated_land_quantities_graph_data = excavated_land_quantities_graph.build()
        if self.computed.excavated_land_quantities_graph_data:
            self.all_rndts_data_empty = False

        excavated_land_statements_graph = RNDTSStatementsGraphProcessor(
            self.siret,
            self.rndts_data["excavated_land_incoming"],
            self.rndts_data["excavated_land_outgoing"],
            "excavated_land",
            data_date_interval,
        )
        self.computed.excavated_land_statements_graph_data = excavated_land_statements_graph.build()
        if self.computed.excavated_land_statements_graph_data:
            self.all_rndts_data_empty = False

        ssd_quantities_graph = RNDTSQuantitiesGraphProcessor(
            self.siret, None, self.rndts_data["ssd_data"], data_date_interval
        )
        self.computed.ssd_quantities_graph_data = ssd_quantities_graph.build()
        if self.computed.ssd_quantities_graph_data:
            self.all_rndts_data_empty = False

        ssd_statements_graph = RNDTSStatementsGraphProcessor(
            self.siret,
            None,
            self.rndts_data["ssd_data"],
            "ssd",
            data_date_interval,
        )
        self.computed.ssd_statements_graph_data = ssd_statements_graph.build()
        if self.computed.ssd_statements_graph_data:
            self.all_rndts_data_empty = False

        rndts_transporter_statements_stats_graph = RNDTSTransporterStatementsStatsGraphProcessor(
            self.siret, self.rndts_data, data_date_interval
        )
        self.computed.rndts_transporter_statement_stats_graph_data = rndts_transporter_statements_stats_graph.build()
        if self.computed.rndts_transporter_statement_stats_graph_data:
            self.all_rndts_data_empty = False

        rndts_transporter_quantities_graph = RNDTSTransporterQuantitiesGraphProcessor(
            self.siret, self.rndts_data, data_date_interval
        )
        self.computed.rndts_transporter_quantities_graph_data = rndts_transporter_quantities_graph.build()
        if self.computed.rndts_transporter_quantities_graph_data:
            self.all_rndts_data_empty = False

        eco_organisme_bordereaux_graph = IntermediaryBordereauxCountsGraphProcessor(
            company_siret=self.siret,
            bs_data_dfs={k: v for k, v in self.bs_dfs.items() if k in [BSDD, BSDD_NON_DANGEROUS, BSDA, BSDASRI]},
            transporters_data_df=self.transporter_data_dfs,
            data_date_interval=data_date_interval,
        )
        self.computed.eco_organisme_bordereaux_graph_data = eco_organisme_bordereaux_graph.build()
        if self.computed.eco_organisme_bordereaux_graph_data:
            self.all_bsd_data_empty = False

        eco_organisme_quantities_graph = IntermediaryBordereauxQuantitiesGraphProcessor(
            company_siret=self.siret,
            bs_data_dfs={k: v for k, v in self.bs_dfs.items() if k in [BSDD, BSDD_NON_DANGEROUS, BSDA, BSDASRI]},
            transporters_data_df=self.transporter_data_dfs,
            data_date_interval=data_date_interval,
        )
        self.computed.eco_organisme_quantities_graph_data = eco_organisme_quantities_graph.build()
        if self.computed.eco_organisme_bordereaux_graph_data:
            self.all_bsd_data_empty = False

    def _build_html_components(self):
        data_date_interval = (self.data_start_date, self.data_end_date)

        for bs_type, df in self.bs_dfs.items():
            quantity_variables = get_quantity_variable_names(bs_type)
            packaging_data = None
            if bs_type == BSFF:
                packaging_data = self.bsff_packagings_df

            bs_stats = BsdStatsProcessor(
                self.siret,
                bs_type,
                df,
                data_date_interval,
                quantity_variables_names=quantity_variables,
                bs_revised_data=self.revised_bs_dfs.get(bs_type, None),
                packagings_data=packaging_data,
            )
            bs_stats_data = bs_stats.build()
            if bs_stats_data:
                self.all_bsd_data_empty = False
            setattr(self.computed, f"{bs_type}_stats_data", bs_stats_data)

        agreement_data = ReceiptAgrementsProcessor(self.receipts_agreements_data)
        self.computed.agreement_data = agreement_data.build()

        linked_companies_data = LinkedCompaniesProcessor(
            company_siret=self.siret,
            linked_companies_data=self.linked_companies_data,
        )
        self.computed.linked_companies_data = linked_companies_data.build()

        icpe_processor = ICPEItemsProcessor(
            self.computed.org_id,
            self.icpe_data,
        )
        self.computed.icpe_data = icpe_processor.build()

        waste_flows_table = WasteFlowsTableProcessor(
            self.siret,
            self.bs_dfs,
            self.transporter_data_dfs,
            self.rndts_data,
            data_date_interval,
            WASTE_CODES_DATA,
            self.bsff_packagings_df,
        )
        self.computed.waste_flows_data = waste_flows_table.build()

        storage_stats = StorageStatsProcessor(
            self.siret,
            self.bs_dfs,
            WASTE_CODES_DATA,
            data_date_interval,
        )
        self.computed.storage_data = storage_stats.build()

        traceability_interruptions = TraceabilityInterruptionsProcessor(
            self.siret,
            self.bs_dfs[BSDD],
            WASTE_CODES_DATA,
            data_date_interval,
        )
        self.computed.traceability_interruptions_data = traceability_interruptions.build()

        waste_is_dangerous_statements = WasteIsDangerousStatementsProcessor(
            self.siret,
            self.bs_dfs[BSDD],
            self.transporter_data_dfs.get(BSDD),
            WASTE_CODES_DATA,
            data_date_interval,
        )
        self.computed.waste_is_dangerous_statements_data = waste_is_dangerous_statements.build()

        bsd_canceled_table = BsdCanceledTableProcessor(
            self.siret,
            self.bs_dfs,
            self.revised_bs_dfs,
            data_date_interval,
        )
        self.computed.bsd_canceled_data = bsd_canceled_table.build()

        same_emitter_recipient_table = SameEmitterRecipientTableProcessor(
            self.bs_dfs,
            self.transporter_data_dfs,
            data_date_interval,
        )
        self.computed.same_emitter_recipient_data = same_emitter_recipient_table.build()

        private_individuals_collections_table = PrivateIndividualsCollectionsTableProcessor(
            self.siret,
            self.bs_dfs[BSDA],
            self.transporter_data_dfs.get(BSDA),
            data_date_interval,
        )
        self.computed.private_individuals_collections_data = private_individuals_collections_table.build()

        quantity_outliers_table = QuantityOutliersTableProcessor(
            self.bs_dfs,
            self.transporter_data_dfs,
            data_date_interval,
        )
        self.computed.quantity_outliers_data = quantity_outliers_table.build()

        waste_processing_without_icpe_data = WasteProcessingWithoutICPERubriqueProcessor(
            self.siret, self.bs_dfs, self.rndts_data["ndw_incoming"], self.icpe_data, data_date_interval
        )
        self.computed.bs_processed_without_icpe_authorization = waste_processing_without_icpe_data.build()

        bsda_worker_stats = BsdaWorkerStatsProcessor(
            company_siret=self.siret,
            bsda_data_df=self.bs_dfs[BSDA],
            data_date_interval=data_date_interval,
        )
        self.computed.bsda_worker_stats_data = bsda_worker_stats.build()
        if self.computed.bsda_worker_stats_data:
            self.all_bsd_data_empty = False

        bsda_worker_quantities = BsdaWorkerQuantityProcessor(
            company_siret=self.siret,
            bsda_data_df=self.bs_dfs[BSDA],
            bsda_transporters_data_df=self.transporter_data_dfs.get(BSDA),
            data_date_interval=data_date_interval,
        )
        self.computed.bsda_worker_quantity_data = bsda_worker_quantities.build()

        transporter_bordereaux_stats = TransporterBordereauxStatsProcessor(
            company_siret=self.siret,
            transporters_data_df=self.transporter_data_dfs,
            bs_data_dfs={k: v for k, v in self.bs_dfs.items() if k not in BS_TYPES_WITH_MULTIMODAL_TRANSPORT},
            data_date_interval=data_date_interval,
            packagings_data_df=self.bsff_packagings_df,
        )
        self.computed.transporter_bordereaux_stats_data = transporter_bordereaux_stats.build()
        if self.computed.transporter_bordereaux_stats_data:
            self.all_bsd_data_empty = False

        followed_with_pnttd = FollowedWithPNTTDTableProcessor(
            company_siret=self.siret,
            bs_data_dfs={k: v for k, v in self.bs_dfs.items() if k in [BSDD, BSDD_NON_DANGEROUS]},
            data_date_interval=data_date_interval,
            waste_codes_df=WASTE_CODES_DATA,
        )
        self.computed.followed_with_pnttd_data = followed_with_pnttd.build()

        gistrid_data = get_gistrid_data(self.siret)
        gistrid_stats = GistridStatsProcessor(
            company_siret=self.siret,
            gistrid_data_df=gistrid_data,
        )
        self.computed.gistrid_stats_data = gistrid_stats.build()

        non_dangerous_waste_stats = RNDTSStatsProcessor(
            self.siret, self.rndts_data["ndw_incoming"], self.rndts_data["ndw_outgoing"], data_date_interval
        )
        self.computed.non_dangerous_waste_stats_data = non_dangerous_waste_stats.build()
        if self.computed.non_dangerous_waste_stats_data:
            self.all_rndts_data_empty = False

        excavated_land_stats = RNDTSStatsProcessor(
            self.siret,
            self.rndts_data["excavated_land_incoming"],
            self.rndts_data["excavated_land_outgoing"],
            data_date_interval,
        )
        self.computed.excavated_land_stats_data = excavated_land_stats.build()
        if self.computed.excavated_land_stats_data:
            self.all_rndts_data_empty = False

        ssd_stats = RNDTSStatsProcessor(self.siret, None, self.rndts_data["ssd_data"], data_date_interval)
        self.computed.ssd_stats_data = ssd_stats.build()
        if self.computed.ssd_stats_data:
            self.all_rndts_data_empty = False

        ssd_table = SSDProcessor(self.siret, self.rndts_data["ssd_data"], data_date_interval)
        self.computed.ssd_table_data = ssd_table.build()
        if self.computed.ssd_table_data:
            self.all_rndts_data_empty = False

        rndts_transporter_stats = RNDTSTransporterStatsProcessor(self.siret, self.rndts_data, data_date_interval)
        self.computed.rndts_transporter_stats_data = rndts_transporter_stats.build()
        if self.computed.rndts_transporter_stats_data:
            self.all_rndts_data_empty = False

        eco_organisme_bordereaux_stats = IntermediaryBordereauxStatsProcessor(
            company_siret=self.siret,
            bs_data_dfs={k: v for k, v in self.bs_dfs.items() if k in [BSDD, BSDD_NON_DANGEROUS, BSDA, BSDASRI]},
            transporters_data_df=self.transporter_data_dfs,
            data_date_interval=data_date_interval,
        )
        self.computed.eco_organisme_bordereaux_stats_data = eco_organisme_bordereaux_stats.build()
        if self.computed.eco_organisme_bordereaux_graph_data:
            self.all_bsd_data_empty = False

        incinerator_outgoing_waste_data = IncineratorOutgoingWasteProcessor(
            self.siret,
            self.bs_dfs,
            self.transporter_data_dfs,
            self.icpe_data,
            self.rndts_data["ndw_outgoing"],
            data_date_interval,
        )
        self.computed.incinerator_outgoing_waste_data = incinerator_outgoing_waste_data.build()

    def process(self):
        self.computed.processing_start = timezone.now()
        self.computed.save()

        # Extract all datasets that will be needed to build the components
        self._extract_data()

        # Get needed company attributes from extracted data
        self._process_company_data()

        # Build all components
        self._build_components()

        self.computed.all_bsd_data_empty = self.all_bsd_data_empty
        self.computed.all_rndts_data_empty = self.all_rndts_data_empty

        self.computed.processing_end = timezone.now()

        self.computed.state = ComputedInspectionData.StateChoice.COMPUTED
        self.computed.save()

        ComputedInspectionData.objects.mark_as_computed(self.computed.pk)
