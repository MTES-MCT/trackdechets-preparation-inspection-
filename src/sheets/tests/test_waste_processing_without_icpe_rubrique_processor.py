from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas import Timestamp

from sheets.constants import BSDA, BSDD

from ..graph_processors.html_components_processors import WasteProcessingWithoutICPERubriqueProcessor


# Example data for testing
@pytest.fixture
def sample_data():
    # Sample data for ICPE data with different "rubriques"
    icpe_data = pd.DataFrame({"rubrique": ["2770", "2791-1", "2718-1"]})

    # Sample bordereau data for BSDD
    bsdd_data = pd.DataFrame(
        {
            "recipient_company_siret": ["12345678900000", "12345678900000", "12345678900000"],
            "processing_operation_code": ["D8", "D5", "D10"],
            "processed_at": [
                datetime(2023, 1, 1),
                datetime(2023, 2, 2),
                datetime(2023, 4, 3),
            ],
            "quantity_received": [12.5, 22, 30.1],
            "quantity_refused": [0, 2, 25.1],
        }
    )

    # Sample bordereau data for BSDA
    bsda_data = pd.DataFrame(
        {
            "recipient_company_siret": ["12345678900000", "12345678900000"],
            "processing_operation_code": ["D5", "R2"],
            "processed_at": [
                datetime(2023, 2, 2),
                datetime(2023, 4, 3),
            ],
            "quantity_received": [5, 15],
        }
    )

    # Sample RNDTS incoming data
    rndts_incoming_data = pd.DataFrame(
        {
            "etablissement_numero_identification": ["12345678900000", "12345678900000"],
            "date_reception": [
                datetime(2023, 1, 1),
                datetime(2023, 2, 2),
            ],
            "code_traitement": ["D5", "R1"],
            "quantite": [30, 20],
        }
    )

    bs_data_dfs = {BSDD: bsdd_data, BSDA: bsda_data}
    data_date_interval = (
        datetime(2023, 1, 1),
        datetime(2023, 4, 30),
    )

    return bs_data_dfs, rndts_incoming_data, icpe_data, data_date_interval


def test_initialization(sample_data):
    """Test that the processor initializes with correct parameters."""

    bs_data_dfs, rndts_incoming_data, icpe_data, data_date_interval = sample_data
    processor = WasteProcessingWithoutICPERubriqueProcessor(
        company_siret="12345678900000",
        bs_data_dfs=bs_data_dfs,
        rndts_incoming_data=rndts_incoming_data,
        icpe_data=icpe_data,
        data_date_interval=data_date_interval,
        packagings_data_df=None,
    )
    assert processor.siret == "12345678900000"
    assert processor.data_date_interval is not None
    assert isinstance(processor.bs_data_dfs, dict)
    assert isinstance(processor.icpe_data, pd.DataFrame)


def test_preprocess_data_multi_rubriques(sample_data):
    """Test that multiple rubrique processing works correctly."""
    bs_data_dfs, rndts_incoming_data, icpe_data, data_date_interval = sample_data
    processor = WasteProcessingWithoutICPERubriqueProcessor(
        company_siret="12345678900000",
        bs_data_dfs=bs_data_dfs,
        rndts_incoming_data=rndts_incoming_data,
        icpe_data=icpe_data,
        data_date_interval=data_date_interval,
        packagings_data_df=None,
    )

    processor._preprocess_data_multi_rubriques()
    preprocessed_data = processor.preprocessed_data["dangerous"]
    assert len(preprocessed_data) == 1

    item = preprocessed_data[0]
    expected_output = {
        "missing_rubriques": "2760-1, 2760-2",
        "num_missing_rubriques": 2,
        "found_processing_codes": "D5",
        "num_found_processing_codes": 1,
        "bs_list": pd.DataFrame(
            {
                "recipient_company_siret": ["12345678900000", "12345678900000"],
                "processing_operation_code": ["D5", "D5"],
                "processed_at": [Timestamp("2023-02-02 00:00:00"), Timestamp("2023-02-02 00:00:00")],
                "quantity_received": [22.0, 5.0],
                "quantity_refused": [2.0, np.nan],
                "bs_type": ["BSDD", "BSDA"],
            }
        ),
        "stats": {"total_bs": "2", "total_quantity": "25"},
    }

    assert item.keys() == expected_output.keys()

    for key in item.keys():
        if key == "bs_list":
            # Dataframes equality test
            assert item[key].equals(expected_output[key])
        else:
            assert item[key] == expected_output[key]


def test_preprocess_data_single_rubrique(sample_data):
    """Test that single rubrique processing works correctly."""
    bs_data_dfs, rndts_incoming_data, icpe_data, data_date_interval = sample_data
    processor = WasteProcessingWithoutICPERubriqueProcessor(
        company_siret="12345678900000",
        bs_data_dfs=bs_data_dfs,
        rndts_incoming_data=rndts_incoming_data,
        icpe_data=icpe_data,
        data_date_interval=data_date_interval,
        packagings_data_df=None,
    )

    processor._preprocess_data_single_rubrique()
    dangerous_data = processor.preprocessed_data["dangerous"]
    assert len(dangerous_data) == 1

    item = dangerous_data[0]

    expected_output = {
        "missing_rubriques": "2790",
        "num_missing_rubriques": 1,
        "found_processing_codes": "R2, D8",
        "num_found_processing_codes": 2,
        "bs_list": pd.DataFrame(
            {
                "recipient_company_siret": ["12345678900000", "12345678900000"],
                "processing_operation_code": ["R2", "D8"],
                "processed_at": [Timestamp("2023-04-03 00:00:00"), Timestamp("2023-01-01 00:00:00")],
                "quantity_received": [15.0, 12.5],
                "quantity_refused": [np.nan, 0.0],
                "bs_type": ["BSDA", "BSDD"],
            }
        ),
        "stats": {"total_bs": "2", "total_quantity": "27.5"},
    }

    assert item.keys() == expected_output.keys()
    for key in item.keys():
        if key == "bs_list":
            # Dataframes equality test
            assert item[key].equals(expected_output[key])
        else:
            assert item[key] == expected_output[key]


def test_preprocess_non_dangerous_rubriques(sample_data):
    """Test that non-dangerous rubrique processing works correctly."""
    bs_data_dfs, rndts_incoming_data, icpe_data, data_date_interval = sample_data
    processor = WasteProcessingWithoutICPERubriqueProcessor(
        company_siret="12345678900000",
        bs_data_dfs=bs_data_dfs,
        rndts_incoming_data=rndts_incoming_data,
        icpe_data=icpe_data,
        data_date_interval=data_date_interval,
        packagings_data_df=None,
    )
    processor._preprocess_non_dangerous_rubriques()
    non_dangerous_data = processor.preprocessed_data["non_dangerous"]

    assert len(non_dangerous_data) == 2

    expected_output = [
        {
            "missing_rubriques": "2760-2",
            "num_missing_rubriques": 1,
            "found_processing_codes": "D5",
            "num_found_processing_codes": 1,
            "statements_list": pd.DataFrame(
                {
                    "siret": ["12345678900000"],
                    "date_reception": [Timestamp("2023-01-01 00:00:00")],
                    "code_traitement": ["D5"],
                    "quantite": [30],
                }
            ),
            "stats": {"total_statements": "1", "total_quantity": "30"},
        },
        {
            "missing_rubriques": "2771",
            "num_missing_rubriques": 1,
            "found_processing_codes": "R1",
            "num_found_processing_codes": 1,
            "statements_list": pd.DataFrame(
                {
                    "siret": ["12345678900000"],
                    "date_reception": [Timestamp("2023-02-02 00:00:00")],
                    "code_traitement": ["R1"],
                    "quantite": [20],
                }
            ),
            "stats": {"total_statements": "1", "total_quantity": "20"},
        },
    ]

    for item, expected_item in zip(non_dangerous_data, expected_output):
        assert item.keys() == expected_item.keys()

        for key in item.keys():
            if key == "statements_list":
                # Dataframes equality test
                assert item[key].equals(expected_item[key])
            else:
                assert item[key] == expected_item[key]


def test_check_data_empty():
    """Test that the data check function works correctly."""

    processor = WasteProcessingWithoutICPERubriqueProcessor(
        company_siret="12345678900000",
        bs_data_dfs={BSDD: None, BSDA: pd.DataFrame()},
        rndts_incoming_data=None,
        icpe_data=None,
        data_date_interval=(
            datetime(2023, 1, 1),
            datetime(2023, 4, 30),
        ),
        packagings_data_df=None,
    )
    # Trigger data preprocessing
    processor._preprocess_data()
    assert processor._check_data_empty() is True
