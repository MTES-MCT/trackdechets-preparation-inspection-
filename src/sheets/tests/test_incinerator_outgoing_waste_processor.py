import json
from datetime import datetime

import pandas as pd
import pytest

from sheets.constants import BSDA, BSDASRI, BSDD

from ..graph_processors.html_components_processors import IncineratorOutgoingWasteProcessor
from .constants import EXPECTED_FILES_PATH


@pytest.fixture
def sample_data() -> dict:
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
                "waste_name": ["Déchet A", None, "Déchet B", "Déchet B"],
                "processing_operation_code": ["D10", "D10", "D5", "R1"],
                "quantity_received": [10, 20, 30, 19],
                "quantity_refused": [None, 0, 18, 5],
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
                "waste_name": ["Déchet C", "Déchet D", "Déchet B", "Déchet C"],
                "processing_operation_code": ["D10", "D10", "D5", "R1"],
                "quantity_received": [12.5, 32, 9.3, 10],
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
                "quantity_refused": [None, 0, 18, 5],
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
    }

    rndts_data_df = {
        "ndw_outgoing": pd.DataFrame(
            {
                "producteur_numero_identification": ["12345678901234", "98765432109876", "12345678901234"],
                "destinataire_numero_identification": ["98765432109876", "98765432109876", "56789012345678"],
                "date_expedition": [datetime(2023, 2, 1), datetime(2023, 4, 20), datetime(2023, 6, 10)],
                "quantite": [15, 25, 35],
                "code_dechet": ["04 01 01", "04 01 02", "02 01 03"],
                "denomination_usuelle": ["Déchet AA", "Déchet BB", None],
                "unite": ["M3", "T", "T"],
                "numeros_indentification_transporteurs": [
                    ["98765432109876"],
                    ["12345678901234", "98765432109876"],
                    ["98765432109876"],
                ],
                "code_traitement": ["R1", "R2", "D5"],
            }
        )
    }

    icpe_data_df = pd.DataFrame({"rubrique": ["2770", "2771"]})

    return {
        "bs_data": bs_data_dfs,
        "transporters_data": transporters_data_df,
        "rndts_data": rndts_data_df,
        "icpe_data": icpe_data_df,
    }


@pytest.fixture
def sample_data_date_interval():
    """Sample data date interval for filtering"""
    return (datetime(2023, 1, 1), datetime(2023, 6, 30))


def test_preprocess_bs_data(sample_data, sample_data_date_interval):
    """Test preprocessing of BS data for dangerous waste"""
    processor = IncineratorOutgoingWasteProcessor(
        company_siret="12345678901234",
        bs_data_dfs=sample_data["bs_data"],
        transporters_data_df=sample_data["transporters_data"],
        icpe_data=sample_data["icpe_data"],
        rndts_outgoing_data=None,  # Testing only BS data
        data_date_interval=sample_data_date_interval,
    )
    processor._preprocess_bs_data()

    preprocessed_data = processor.preprocessed_data["dangerous"]
    assert len(preprocessed_data) == 5

    expected_data = pd.DataFrame(
        [
            {
                "waste_code": "01 01 01*",
                "recipient_company_siret": "43210987654321",
                "processing_operation_code": "D10",
                "quantity": 30.0,
                "waste_name": "Déchet A",
            },
            {
                "waste_code": "01 01 03*",
                "recipient_company_siret": "87654321098765",
                "processing_operation_code": "D5",
                "quantity": 9.3,
                "waste_name": "Déchet B",
            },
            {
                "waste_code": "02 01 01*",
                "recipient_company_siret": "87654321098765",
                "processing_operation_code": "D10",
                "quantity": 12.5,
                "waste_name": "Déchet C",
            },
            {
                "waste_code": "04 01 01*",
                "recipient_company_siret": "43210987654321",
                "processing_operation_code": "D10",
                "quantity": 3.1799999999999997,
                "waste_name": "Déchet A",
            },
            {
                "waste_code": "04 01 03*",
                "recipient_company_siret": "87654321098769",
                "processing_operation_code": "R1",
                "quantity": 19.0,
                "waste_name": "Déchet B",
            },
        ]
    )

    assert preprocessed_data.equals(expected_data)


def test_preprocess_rndts_statements_data(sample_data, sample_data_date_interval):
    """Test preprocessing of RNDTS data for non-dangerous waste"""
    processor = IncineratorOutgoingWasteProcessor(
        company_siret="12345678901234",
        bs_data_dfs={},
        transporters_data_df={},
        icpe_data=sample_data["icpe_data"],
        rndts_outgoing_data=sample_data["rndts_data"]["ndw_outgoing"],
        data_date_interval=sample_data_date_interval,
    )
    processor._preprocess_rndts_statements_data()

    preprocessed_data = processor.preprocessed_data["non_dangerous"]
    assert len(preprocessed_data) == 2

    expected_data = pd.DataFrame(
        [
            {
                "code_dechet": "02 01 03",
                "destinataire_numero_identification": "56789012345678",
                "code_traitement": "D5",
                "unite": "T",
                "quantite": 35,
                "denomination_usuelle": "",
            },
            {
                "code_dechet": "04 01 01",
                "destinataire_numero_identification": "98765432109876",
                "code_traitement": "R1",
                "unite": "M3",
                "quantite": 15,
                "denomination_usuelle": "Déchet AA",
            },
        ]
    )

    assert preprocessed_data.equals(expected_data)


def test_is_incinerator(sample_data):
    """Test the incinerator detection method"""
    processor = IncineratorOutgoingWasteProcessor(
        company_siret="12345678901234",
        bs_data_dfs={},
        transporters_data_df={},
        icpe_data=sample_data["icpe_data"],
        rndts_outgoing_data=None,
        data_date_interval=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
    )
    assert processor.is_incinerator(dangerous_waste=True), "Incinerator detection failed for dangerous waste"
    assert processor.is_incinerator(dangerous_waste=False), "Incinerator detection failed for non-dangerous waste"


def test_build_with_no_data(sample_data):
    """Test build method when no data is available"""
    processor = IncineratorOutgoingWasteProcessor(
        company_siret="87654321098765",
        bs_data_dfs={},
        transporters_data_df={},
        icpe_data=sample_data["icpe_data"],
        rndts_outgoing_data=None,
        data_date_interval=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
    )
    result = processor.build()
    assert result == {}, "Expected empty result when no data is available"
    assert processor._check_data_empty()


def test_build_with_data(
    sample_data,
    sample_data_date_interval,
):
    """Test build method with both BS and RNDTS data"""
    processor = IncineratorOutgoingWasteProcessor(
        company_siret="12345678901234",
        bs_data_dfs=sample_data["bs_data"],
        transporters_data_df=sample_data["transporters_data"],
        icpe_data=sample_data["icpe_data"],
        rndts_outgoing_data=sample_data["rndts_data"]["ndw_outgoing"],  # Testing only BS data
        data_date_interval=sample_data_date_interval,
    )
    result = processor.build()
    assert len(result["dangerous"]) > 0, "No dangerous waste data in result"
    assert len(result["non_dangerous"]) > 0, "No non-dangerous waste data in result"

    expected_data = json.load(
        (EXPECTED_FILES_PATH / "incinerator_outgoing_waste_build_data_expected.json").open(encoding="utf-8")
    )

    assert result == expected_data
