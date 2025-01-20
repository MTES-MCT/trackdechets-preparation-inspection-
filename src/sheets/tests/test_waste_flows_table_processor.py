from datetime import datetime

import pandas as pd
import pytest

from sheets.constants import BSDA, BSDASRI, BSDD, BSFF
from sheets.data_extract import load_waste_code_data

from ..graph_processors.html_components_processors import (
    WasteFlowsTableProcessor,
)  # Adjust the import to your actual module
from .constants import EXPECTED_FILES_PATH


@pytest.fixture
def sample_data() -> dict:
    # Creating sample data for bs_data_dfs, transporters_data_df, rndts_data, waste_codes_df, and packagings_data
    bs_data_dfs = {
        BSDD: pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "emitter_company_siret": ["12345678901234", "12345678901234", "98765432109876", "98765432109876"],
                "recipient_company_siret": ["43210987654321", "43210987654321", "12345678901234", "87654321098769"],
                "received_at": [
                    datetime(2023, 1, 11),
                    datetime(2023, 3, 18),
                    datetime(2023, 5, 24),
                    datetime(2023, 5, 21),
                ],
                "waste_code": ["01 01 01*", "01 01 01*", "01 01 03*", "01 01 03*"],
                "quantity_received": [10, 20, 30, 19],
                "quantity_refused": [3, None, 30, 7],
            }
        ),
        BSDA: pd.DataFrame(
            {
                "id": [4, 5, 6, 1],
                "emitter_company_siret": ["12345678901234", "98765432109876", "12345678901234", "87654321098765"],
                "recipient_company_siret": ["87654321098765", "12345678901234", "87654321098765", "97654321098765"],
                "received_at": [
                    datetime(2023, 2, 1),
                    datetime(2023, 4, 20),
                    datetime(2023, 6, 10),
                    datetime(2023, 7, 14),
                ],
                "waste_code": ["02 01 01*", "02 01 02*", "01 01 03*", "02 01 01*"],
                "quantity_received": [12.5, 32, 9.3, 10],
            }
        ),
        BSFF: pd.DataFrame(
            {
                "id": [8, 5, 6, 2],
                "emitter_company_siret": ["12345678901234", "98765432109876", "12345678901234", "87654321098765"],
                "recipient_company_siret": ["87654321098765", "12345678901234", "87654321098765", "97654321098765"],
                "received_at": [
                    datetime(2023, 2, 1),
                    datetime(2023, 4, 20),
                    datetime(2023, 6, 10),
                    datetime(2023, 6, 14),
                ],
                "waste_code": ["03 01 01*", "03 01 02*", "03 01 03*", "03 01 03*"],
            }
        ),
        BSDASRI: pd.DataFrame(
            {
                "id": [11, 22, 33, 44],
                "emitter_company_siret": ["12345678901234", "87654321098765", "98765432109876", "12345678901234"],
                "recipient_company_siret": ["43210987654321", "43210987654321", "12345678901234", "87654321098769"],
                "sent_at": [
                    datetime(2023, 1, 8),
                    datetime(2023, 2, 18),
                    datetime(2023, 5, 11),
                    datetime(2023, 5, 20),
                ],
                "received_at": [
                    datetime(2023, 1, 11),
                    datetime(2023, 3, 18),
                    datetime(2023, 5, 24),
                    datetime(2023, 5, 21),
                ],
                "waste_code": ["04 01 01*", "04 01 01*", "04 01 03*", "04 01 03*"],
                "waste_name": ["Déchet A", None, "Déchet B", "Déchet B"],
                "processing_operation_code": ["D10", "D10", "D5", "R1"],
                "quantity_received": [7.3, 0.02, 0.54, 19],
                "quantity_refused": [4.12, 0.01, 0.32, None],
            }
        ),
    }

    transporters_data_df = {
        BSDD: pd.DataFrame(
            {
                "bs_id": [1, 2, 3, 4],
                "transporter_company_siret": ["56789012345678", "56789012345678", "97654321098765", "12345678901234"],
                "sent_at": [
                    datetime(2023, 1, 10),
                    datetime(2023, 3, 15),
                    datetime(2023, 5, 20),
                    datetime(2023, 5, 20),
                ],
                "quantity_received": [10, 20, 30, 19],
            }
        ),
        BSDA: pd.DataFrame(
            {
                "bs_id": [4, 5, 6, 1],
                "transporter_company_siret": ["56789012345678", "56789012345678", "97654321098765", "12345678901234"],
                "sent_at": [
                    datetime(2023, 1, 10),
                    datetime(2023, 3, 15),
                    datetime(2023, 5, 20),
                    datetime(2023, 5, 20),
                ],
                "quantity_received": [12.5, 32, 9.3, 10],
            }
        ),
        BSFF: pd.DataFrame(
            {
                "bs_id": [8, 5, 6, 2],
                "transporter_company_siret": [
                    "56789012345678",
                    "56789012345678",
                    "12345678901234",
                    "12345678901234",
                ],  # The third is to test the case where the company as emitted and transported the same waste (multiple flow status)
                "sent_at": [
                    datetime(2023, 1, 10),
                    datetime(2023, 3, 15),
                    datetime(2023, 5, 20),
                    datetime(2023, 5, 20),
                ],
            }
        ),
    }

    packagings_data = pd.DataFrame(
        {
            "bsff_id": [8, 8, 5, 6, 2],
            "acceptation_weight": [3, 1.001, 1, 2.4, 5],
            "acceptation_date": [
                datetime(2023, 2, 1),
                datetime(2023, 2, 1),
                datetime(2023, 4, 20),
                datetime(2023, 6, 10),
                datetime(2023, 6, 10),
            ],
        }
    )

    rndts_data = {
        "ndw_incoming": pd.DataFrame(
            {
                "etablissement_numero_identification": ["12345678901234", "12345678901234", "98765432109876"],
                "date_reception": [datetime(2023, 1, 10), datetime(2023, 3, 15), datetime(2023, 5, 20)],
                "quantite": [10, 20, 30],
                "code_dechet": ["03 01 01", "03 01 02", "03 01 01"],
                "unite": ["T", "M3", "T"],
                "numeros_indentification_transporteurs": [["98765432109876"], ["98765432109876"], ["12345678901234"]],
            }
        ),
        "ndw_outgoing": pd.DataFrame(
            {
                "producteur_numero_identification": ["12345678901234", "98765432109876", "12345678901234"],
                "date_expedition": [datetime(2023, 2, 1), datetime(2023, 4, 20), datetime(2023, 6, 10)],
                "quantite": [15, 25, 35],
                "code_dechet": ["04 01 01", "04 01 02", "02 01 03"],
                "unite": ["M3", "T", "T"],
                "numeros_indentification_transporteurs": [
                    ["98765432109876"],
                    ["12345678901234", "98765432109876"],
                    ["98765432109876"],
                ],
            }
        ),
        "excavated_land_incoming": pd.DataFrame(
            {
                "etablissement_numero_identification": ["12345678901234", "12345678901234", "98765432109876"],
                "date_reception": [datetime(2023, 2, 10), datetime(2023, 3, 18), datetime(2023, 5, 21)],
                "quantite": [9.7, 12, 18],
                "code_dechet": ["03 01 01", "03 01 02", "03 01 03"],
                "unite": ["T", "M3", "T"],
                "numeros_indentification_transporteurs": [["98765432109876"], ["98765432109876"], ["12345678901234"]],
            }
        ),
        "excavated_land_outgoing": pd.DataFrame(
            {
                "producteur_numero_identification": ["12345678901234", "98765432109876", "12345678901234"],
                "date_expedition": [datetime(2023, 2, 1), datetime(2023, 2, 20), datetime(2023, 4, 17)],
                "quantite": [15, 25, 35],
                "code_dechet": ["05 01 01", "03 01 02", "02 01 03"],
                "unite": ["M3", "T", "T"],
                "numeros_indentification_transporteurs": [
                    ["98765432109876"],
                    ["12345678901234", "98765432109876"],
                    ["98765432109876"],
                ],
            }
        ),
    }

    return {
        "bs_data_dfs": bs_data_dfs,
        "transporters_data_df": transporters_data_df,
        "rndts_data": rndts_data,
        "packagings_data": packagings_data,
    }


@pytest.fixture
def waste_code_data() -> pd.DataFrame:
    return load_waste_code_data()


def test_preprocess_bs_data(sample_data: dict, waste_code_data: pd.DataFrame):
    processor = WasteFlowsTableProcessor(
        company_siret="12345678901234",
        bs_data_dfs=sample_data["bs_data_dfs"],
        transporters_data_df=sample_data["transporters_data_df"],
        rndts_data=sample_data["rndts_data"],
        data_date_interval=(datetime(2023, 1, 1), datetime(2023, 6, 30)),
        waste_codes_df=waste_code_data,
        packagings_data=sample_data["packagings_data"],
    )

    bs_data = processor._preprocess_bs_data()

    expected_output = pd.DataFrame(
        {
            "waste_code": [
                "01 01 01*",
                "01 01 03*",
                "01 01 03*",
                "01 01 03*",
                "02 01 01*",
                "02 01 01*",
                "02 01 02*",
                "03 01 01*",
                "03 01 02*",
                "03 01 03*",
                "03 01 03*",
                "04 01 01*",
                "04 01 03*",
                "04 01 03*",
            ],
            "flow_status": [
                "outgoing",
                "incoming",
                "outgoing",
                "transported",
                "outgoing",
                "transported",
                "incoming",
                "outgoing",
                "incoming",
                "outgoing",
                "transported",
                "outgoing",
                "incoming",
                "outgoing",
            ],
            "quantity_received": [27.0, 0.0, 9.3, 12.0, 12.5, 10.0, 32.0, 4.001, 1.0, 2.4, 7.4, 7.3, 0.54, 19.0],
            "unit": ["t", "t", "t", "t", "t", "t", "t", "t", "t", "t", "t", "t", "t", "t"],
        }
    )
    assert bs_data is not None
    assert bs_data.equals(expected_output)


def test_preprocess_rndts_data(sample_data: dict, waste_code_data: pd.DataFrame):
    processor = WasteFlowsTableProcessor(
        company_siret="12345678901234",
        bs_data_dfs=sample_data["bs_data_dfs"],
        transporters_data_df=sample_data["transporters_data_df"],
        rndts_data=sample_data["rndts_data"],
        data_date_interval=(datetime(2023, 1, 1), datetime(2023, 6, 30)),
        waste_codes_df=waste_code_data,
        packagings_data=sample_data["packagings_data"],
    )

    preprocessed_rndts_data = processor._preprocess_rndts_data()

    expected_output = pd.DataFrame(
        {
            "waste_code": [
                "03 01 01",
                "03 01 02",
                "03 01 01",
                "02 01 03",
                "04 01 01",
                "04 01 02",
                "03 01 01",
                "03 01 02",
                "03 01 03",
                "02 01 03",
                "05 01 01",
                "03 01 02",
            ],
            "unit": ["t", "m³", "t", "t", "m³", "t", "t", "m³", "t", "t", "m³", "t"],
            "quantity_received": [10.0, 20.0, 30.0, 35.0, 15.0, 25.0, 9.7, 12.0, 18.0, 35.0, 15.0, 25.0],
            "flow_status": [
                "incoming",
                "incoming",
                "transported_incoming",
                "outgoing",
                "outgoing",
                "transported_outgoing",
                "incoming",
                "incoming",
                "transported_incoming",
                "outgoing",
                "outgoing",
                "transported_outgoing",
            ],
        }
    )
    assert preprocessed_rndts_data is not None
    assert preprocessed_rndts_data.equals(expected_output)


def test_preprocess_data(sample_data: dict, waste_code_data: pd.DataFrame):
    processor = WasteFlowsTableProcessor(
        company_siret="12345678901234",
        bs_data_dfs=sample_data["bs_data_dfs"],
        transporters_data_df=sample_data["transporters_data_df"],
        rndts_data=sample_data["rndts_data"],
        data_date_interval=(datetime(2023, 1, 1), datetime(2023, 6, 30)),
        waste_codes_df=waste_code_data,
        packagings_data=sample_data["packagings_data"],
    )

    processor._preprocess_data()
    preprocessed_df = processor.preprocessed_df

    expected_output = pd.read_csv(
        EXPECTED_FILES_PATH / "waste_flow_preprocessed_data_expected.csv",
        keep_default_na=False,
        dtype=str,
        index_col=0,
    )
    assert preprocessed_df is not None
    assert preprocessed_df.equals(expected_output)


def test_empty_data():
    processor = WasteFlowsTableProcessor(
        company_siret="12345678901234",
        bs_data_dfs={},
        transporters_data_df={},
        rndts_data={},
        data_date_interval=(datetime(2023, 1, 1), datetime(2023, 6, 30)),
        waste_codes_df=pd.DataFrame(),
        packagings_data=None,
    )

    processor._preprocess_data()
    preprocessed_df = processor.preprocessed_df

    assert preprocessed_df is None or preprocessed_df.empty


def test_empty_bs_data_with_transport_data(waste_code_data: pd.DataFrame):
    transport_df = pd.DataFrame(
        {
            "id": ["1", "2"],
            "bs_id": ["5", "6"],
            "sent_at": [datetime(2023, 2, 2), datetime(2023, 5, 2)],
            "transporter_company_siret": ["12345678901234", "12345678901234"],
            "transporter_transport_mode": ["ROAD", "ROAD"],
            "quantity_received": [12.5, 11.3],
            "waste_code": ["13 03 10*", "14 06 01*"],
        }
    )
    processor = WasteFlowsTableProcessor(
        company_siret="12345678901234",
        bs_data_dfs={
            BSDD: pd.DataFrame(
                columns=[
                    "id",
                    "created_at",
                    "sent_at",
                    "received_at",
                    "emitter_company_siret",
                    "emitter_company_address",
                    "recipient_company_siret",
                    "waste_detail_quantity",
                    "waste_code",
                    "quantity_received",
                    "status",
                ]
            )
        },
        transporters_data_df={BSDD: transport_df},
        rndts_data={},
        data_date_interval=(datetime(2023, 1, 1), datetime(2023, 6, 30)),
        waste_codes_df=waste_code_data,
        packagings_data=None,
    )

    processor._preprocess_data()
    preprocessed_df = processor.preprocessed_df

    expected_output = pd.DataFrame(
        {
            "waste_code": ["13 03 10*", "14 06 01*"],
            "description": ["autres huiles isolantes et fluides caloporteurs", "chlorofluorocarbones, HCFC, HFC"],
            "flow_status": ["transported", "transported"],
            "quantity_received": ["12.5", "11.3"],
            "unit": ["t", "t"],
        }
    )
    assert preprocessed_df.equals(expected_output)


def test_empty_packagings_data_with_bsff_and_transport_data(sample_data: dict, waste_code_data: pd.DataFrame):
    processor = WasteFlowsTableProcessor(
        company_siret="12345678901234",
        bs_data_dfs={BSFF: sample_data["bs_data_dfs"][BSFF]},
        transporters_data_df={BSFF: sample_data["transporters_data_df"][BSFF]},
        rndts_data={},
        data_date_interval=(datetime(2023, 1, 1), datetime(2023, 6, 30)),
        waste_codes_df=waste_code_data,
        packagings_data=None,
    )

    processor._preprocess_data()
    preprocessed_df = processor.preprocessed_df

    assert preprocessed_df is None


def test_build(sample_data: dict, waste_code_data: pd.DataFrame):
    processor = WasteFlowsTableProcessor(
        company_siret="12345678901234",
        bs_data_dfs=sample_data["bs_data_dfs"],
        transporters_data_df=sample_data["transporters_data_df"],
        rndts_data=sample_data["rndts_data"],
        data_date_interval=(datetime(2023, 1, 1), datetime(2023, 6, 30)),
        waste_codes_df=waste_code_data,
        packagings_data=sample_data["packagings_data"],
    )

    result = processor.build()

    assert isinstance(result, list)
    assert len(result) == 25
    for record in result:
        assert "waste_code" in record
        assert "description" in record
        assert "flow_status" in record
        assert "quantity_received" in record
        assert "unit" in record
