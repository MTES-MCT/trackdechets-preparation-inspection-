from datetime import datetime

import pandas as pd
import pytest

from ..graph_processors.html_components_processors import PrivateIndividualsCollectionsTableProcessor


@pytest.fixture
def sample_data():
    # Provide sample data as DataFrames
    # id 1 and 4 should be processed
    bsda_data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "emitter_is_private_individual": [True, False, True, True, True],
            "recipient_company_siret": [
                "12345678901234",
                "12345678901234",
                "98765432109876",
                "98765432109876",
                "98765432109871",
            ],
            "worker_company_siret": [None, "12345678901234", None, "12345678901234", "12345678901234"],
            "emitter_company_name": ["Company A", "Company B", "Company C", "Company B", "Company D"],
            "emitter_company_address": ["Address A", "Address B", "Address C", "Address B", "Address D"],
            "worksite_name": ["Site A", "Site B", None, "Site B", None],
            "worksite_address": [
                "Address SA",
                "Address SB",
                None,
                "Address SB",
                "Address SD",
            ],
            "waste_code": ["17 06 05*", "17 06 05*", "17 03 01*", "17 03 01*", "17 06 05*"],
            "waste_name": ["Waste A", "Waste A", "Waste B", "Waste B", "Waste A"],
            "quantity_received": [20, 1.6, 32, 11, 7],
            "received_at": [
                datetime(2024, 8, 10),
                datetime(2024, 8, 12),
                datetime(2024, 8, 15),
                datetime(2024, 8, 19),
                datetime(2023, 8, 19),  # Out of range
            ],
        }
    )

    transport_data = pd.DataFrame(
        {
            "bs_id": [1, 2, 3, 4, 5],
            "sent_at": [
                datetime(2024, 8, 9),
                datetime(2024, 8, 11),
                datetime(2024, 8, 14),
                datetime(2024, 8, 14),
                datetime(2023, 8, 15),
            ],
            "transporter_company_siret": [
                "23456789012345",
                "23456789012345",
                "34567890123456",
                "34567890123456",
                "34567890123456",
            ],
        }
    )

    return bsda_data, transport_data


@pytest.fixture
def date_interval():
    return (datetime(2024, 8, 1), datetime(2024, 8, 31))


def test_initialization(sample_data, date_interval):
    bsda_data, transport_data = sample_data
    processor = PrivateIndividualsCollectionsTableProcessor(
        company_siret="12345678901234",
        bsda_data_df=bsda_data,
        bsda_transporters_data_df=transport_data,
        data_date_interval=date_interval,
    )

    assert processor.company_siret == "12345678901234"
    assert processor.bsda_data_df.equals(bsda_data)
    assert processor.bsda_transporters_data_df.equals(transport_data)
    assert processor.data_date_interval == date_interval


def test_empty_data(sample_data, date_interval):
    empty_df = pd.DataFrame()
    processor = PrivateIndividualsCollectionsTableProcessor(
        company_siret="12345678901234",
        bsda_data_df=empty_df,
        bsda_transporters_data_df=None,
        data_date_interval=date_interval,
    )
    assert processor.build() == []

    bsda_data, transport_data = sample_data
    bsda_data = bsda_data[
        ~bsda_data["id"].isin([1, 4])
    ]  # Get only data that will not be taked into account during the processing
    processor = PrivateIndividualsCollectionsTableProcessor(
        company_siret="12345678901234",
        bsda_data_df=bsda_data,
        bsda_transporters_data_df=transport_data,
        data_date_interval=date_interval,
    )
    assert processor.build() == []


def test_data_preprocessing(sample_data, date_interval):
    bsda_data, transport_data = sample_data
    processor = PrivateIndividualsCollectionsTableProcessor(
        company_siret="12345678901234",
        bsda_data_df=bsda_data,
        bsda_transporters_data_df=transport_data,
        data_date_interval=date_interval,
    )

    processor._preprocess_data()

    assert processor.preprocessed_data is not None
    assert len(processor.preprocessed_data) == 2  # Only 2 rows should match the company_siret


def test_build_output(sample_data, date_interval):
    bsda_data, transport_data = sample_data
    processor = PrivateIndividualsCollectionsTableProcessor(
        company_siret="12345678901234",
        bsda_data_df=bsda_data,
        bsda_transporters_data_df=transport_data,
        data_date_interval=date_interval,
    )

    result = processor.build()

    assert len(result) == 2  # Only 2 rows should match both private individual and company_siret
    assert result[0]["id"] == 1
    assert result[0]["quantity"] == 20
    assert result[0]["sent_at"] == "09/08/2024 00:00"  # Converted to string format

    assert result[1]["id"] == 4
    assert result[1]["quantity"] == 11
    assert result[1]["sent_at"] == "14/08/2024 00:00"
