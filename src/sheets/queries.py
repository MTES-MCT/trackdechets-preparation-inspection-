sql_bsdd_query_str = r"""
select
    id,
    readable_id,
    created_at,
    sent_at,
    received_at,
    processed_at,
    emitter_company_siret,
    emitter_company_address,
    recipient_company_siret,
    waste_details_quantity,
    quantity_received,
    waste_details_code as waste_code,
    waste_details_name as waste_name,
    processing_operation_done as processing_operation_code,
    status,
    no_traceability,
    waste_details_pop as waste_pop,
    waste_details_is_dangerous as is_dangerous,
    emitter_work_site_name as worksite_name,
    emitter_work_site_address as worksite_address,
    next_destination_company_siret,
    next_destination_company_name,
    next_destination_company_country,
    next_destination_company_vat_number,
    next_destination_processing_operation
 from
    trusted_zone_trackdechets.bsdd
where
    (emitter_company_siret = :siret
    or recipient_company_siret = :siret)
    and is_deleted = false
    and created_at BETWEEN :data_start_date AND :data_end_date
    and status::text not in ('DRAFT', 'INITIAL')
    and (waste_details_code ~* '.*\*$' or waste_details_pop or waste_details_is_dangerous)
order by
    created_at ASC;
"""

sql_bsdd_non_dangerous_query_str = r"""
select
    id,
    readable_id,
    created_at,
    sent_at,
    received_at,
    processed_at,
    emitter_company_siret,
    emitter_company_address,
    recipient_company_siret,
    waste_details_quantity,
    quantity_received,
    waste_details_code as waste_code,
    waste_details_name as waste_name,
    processing_operation_done as processing_operation_code,
    status,
    no_traceability,
    waste_details_pop as waste_pop,
    waste_details_is_dangerous as is_dangerous,
    emitter_work_site_name as worksite_name,
    emitter_work_site_address as worksite_address,
    next_destination_company_siret,
    next_destination_company_name,
    next_destination_company_country,
    next_destination_company_vat_number,
    next_destination_processing_operation
 from
    trusted_zone_trackdechets.bsdd
where
    (emitter_company_siret = :siret
    or recipient_company_siret = :siret)
    and is_deleted = false
    and created_at BETWEEN :data_start_date AND :data_end_date
    and status::text not in ('DRAFT', 'INITIAL')
    and not (waste_details_code ~* '.*\*$' or waste_details_pop or waste_details_is_dangerous)
order by
    created_at ASC;
"""

sql_bsdd_transporter_query_str = r"""
select
    id,
    form_id,
    taken_over_at as sent_at,
    transporter_company_siret,
    transporter_number_plate,
    transporter_transport_mode,
    quantity_received,
    waste_details_code as waste_code
from
    refined_zone_enriched.bsdd_transporter_enriched
where
    (emitter_company_siret = :siret
        or recipient_company_siret = :siret
        or transporter_company_siret = :siret
    )
    and bordereau_created_at between :data_start_date and :data_end_date
    and (waste_details_code like '%*'
        or waste_details_pop
        or waste_details_is_dangerous)
"""


sql_bsdd_non_dangerous_transporter_query_str = r"""
select
    id,
    form_id,
    taken_over_at as sent_at,
    transporter_company_siret,
    transporter_number_plate,
    transporter_transport_mode,
    quantity_received,
    waste_details_code as waste_code
from
    refined_zone_enriched.bsdd_transporter_enriched
where
    (emitter_company_siret = :siret
        or recipient_company_siret = :siret
        or transporter_company_siret = :siret
    )
    and bordereau_created_at between :data_start_date and :data_end_date
    and not (waste_details_code like '%*'
        or waste_details_pop
        or waste_details_is_dangerous)
"""

sql_company_query_str = """
select id,
    created_at,
    siret,
    name,
    address,
    company_types,
    transporter_receipt_id,
    trader_receipt_id,
    eco_organisme_agreements,
    broker_receipt_id,
    vhu_agrement_demolisseur_id,
    vhu_agrement_broyeur_id
from trusted_zone_trackdechets.company c
where c.siret = :siret
"""

sql_bsda_query_str = """
select
    id,
    created_at,
    emitter_emission_signature_date,
    worker_work_signature_date,
    transporter_transport_signature_date,
    transporter_transport_taken_over_at as sent_at,
    destination_reception_date as received_at,
    destination_operation_date as processed_at,
    emitter_company_siret,
    emitter_company_name,
    emitter_company_address,
    destination_company_siret as recipient_company_siret,
    weight_value as waste_details_quantity,
    destination_reception_weight as quantity_received,
    waste_code,
    waste_material_name as waste_name,
    destination_operation_code as processing_operation_code,
    status,
    transporter_transport_mode,
    waste_pop,
    emitter_pickup_site_name as worksite_name,
    emitter_pickup_site_address as worksite_address,
    worker_company_siret,
    emitter_is_private_individual,
    transporter_company_siret
from
    trusted_zone_trackdechets.bsda
where
    (emitter_company_siret = :siret
        or destination_company_siret = :siret
        or worker_company_siret = :siret
        or transporter_company_siret = :siret)
    and is_deleted = false
    and created_at BETWEEN :data_start_date AND :data_end_date
    and status::text not in ('DRAFT', 'INITIAL')
    and not is_draft
order by
    created_at asc"""

sql_bsdasri_query_str = """
select
   id,
    created_at,
    transporter_taken_over_at as sent_at,
    destination_reception_date as received_at,
    destination_operation_signature_date as processed_at,
    emitter_company_siret,
    emitter_company_address,
    destination_company_siret as recipient_company_siret,
    emitter_waste_weight_value as waste_details_quantity,
    destination_reception_waste_weight_value as quantity_received,
    destination_reception_waste_volume as volume,
    waste_code,
    destination_operation_code as processing_operation_code,
    status,
    transporter_transport_mode,
    transporter_company_siret
from
        trusted_zone_trackdechets.bsdasri
where
    (emitter_company_siret = :siret
        or destination_company_siret = :siret
        or transporter_company_siret = :siret)
    and is_deleted = false
    and created_at BETWEEN :data_start_date AND :data_end_date
    and status::text not in ('DRAFT', 'INITIAL')
    and not is_draft
order by
    created_at asc"""


sql_bsff_query_str = """select
    id,
    created_at,
    transporter_transport_taken_over_at as sent_at,
    destination_reception_date as received_at,
    emitter_company_siret,
    emitter_company_address,
    destination_company_siret as recipient_company_siret,
    weight_value as waste_detail_quantity,
    waste_code,
    status,
    transporter_transport_mode,
    transporter_company_siret
from
    trusted_zone_trackdechets.bsff
where
    (emitter_company_siret = :siret
        or destination_company_siret = :siret
        or transporter_company_siret = :siret)
    and is_deleted = false
    and created_at BETWEEN :data_start_date AND :data_end_date
    and status::text not in ('DRAFT', 'INITIAL')
    and not is_draft
order by
    created_at asc"""

sql_bsff_packagings_query_str = """
select
    id,
    bsff_id,
    acceptation_status,
    acceptation_date,
    operation_date,
    operation_code,
    volume,
    acceptation_weight,
    weight
from
    trusted_zone_trackdechets.bsff_packaging bp
where
    bp.bsff_id in (
    select
        id
    from
        trusted_zone_trackdechets.bsff
    where
        (emitter_company_siret = :siret
            or destination_company_siret = :siret)
        and is_deleted = false
        and created_at BETWEEN :data_start_date AND :data_end_date
        and status::text not in ('DRAFT', 'INITIAL')
            and not is_draft
        order by
            created_at asc
)
"""

sql_bsvhu_query_str = """
select
    id,
    created_at,
    transporter_transport_taken_over_at as sent_at,
    destination_reception_date as received_at,
    destination_operation_signature_date as processed_at,
    emitter_company_siret,
    emitter_company_address,
    destination_company_siret as recipient_company_siret,
    weight_value as waste_detail_quantity,
    destination_reception_weight as quantity_received,
    waste_code,
    destination_operation_code as processing_operation_code,
    status,
    transporter_company_siret
from
    trusted_zone_trackdechets.bsvhu
where
    (emitter_company_siret = :siret
        or destination_company_siret = :siret
        or transporter_company_siret = :siret)
    and is_deleted = false
    and created_at BETWEEN :data_start_date AND :data_end_date
    and  status::text not in ('DRAFT', 'INITIAL')
    and not is_draft
order by
    created_at asc
"""

sql_revised_bsdd_query_str = """
select id,
    bsdd_id as bs_id,
    created_at,
    updated_at,
    comment,
    is_canceled
from trusted_zone_trackdechets.bsdd_revision_request
where authoring_company_id = :company_id
    and status='ACCEPTED'
    and created_at BETWEEN :data_start_date AND :data_end_date
"""

sql_revised_bsda_query_str = """
select id,
    bsda_id as bs_id,
    created_at,
    updated_at,
    comment,
    is_canceled
from trusted_zone_trackdechets.bsda_revision_request
where authoring_company_id = :company_id
    and status='ACCEPTED'
    and created_at BETWEEN :data_start_date AND :data_end_date
"""

sql_get_icpe_data = """
select
    code_aiot,
    rubrique,
    alinea,
    quantite_totale as quantite,
    unite,
    nature
from
    refined_zone_icpe.installations_rubriques
where siret = :siret
and (regime_rubrique in ('Enregistrement','Déclaration avec contrôle','Déclaration','Autorisation') or regime_rubrique is null)
"""

sql_get_trader_receipt_id_data = """
SELECT id,
    receipt_number,
    validity_limit
FROM trusted_zone_trackdechets.trader_receipt
WHERE id = :id
"""

sql_get_transporter_receipt_id_data_str = """
SELECT id,
    receipt_number,
    validity_limit,
    department
FROM trusted_zone_trackdechets.transporter_receipt
WHERE id = :id
"""

sql_get_broker_receipt_id_data = """
SELECT id,
    receipt_number,
    validity_limit,
    department
FROM trusted_zone_trackdechets.broker_receipt
where id = :id
"""

sql_get_vhu_agrement_data = """
SELECT
    id,
    agrement_number AS receipt_number,
    department
FROM
    trusted_zone_trackdechets.vhu_agrement
WHERE
    id = :id
"""

sql_get_icpe_item_data = """
SELECT
    day_of_processing,
    quantite_traitee AS processed_quantity,
    quantite_autorisee AS authorized_quantity
FROM
    refined_zone_icpe.installations_daily_processed_wastes
WHERE
    siret = :siret
    and rubrique = :rubrique
    and day_of_processing>='2022-01-01'
"""

sql_get_linked_companies_data = """
select
    siret,
    created_at,
    name,
    address
from
    trusted_zone_trackdechets.company c
where
    substring(c.siret for 9) = substring(:siret for 9)
"""

sql_get_gistrid_data_data = """
select
    n.numero_notification,
    n.type_dossier,
    n.numero_gistrid_notifiant,
    n.siret_notifiant,
    n.nom_notifiant,
    n.pays_notifiant,
    n.numero_gistrid_installation_traitement,
    n.siret_installation_traitement,
    n.nom_installation_traitement,
    n.pays_installation_traitement,
    n.somme_quantites_recues,
    n.nombre_transferts_receptionnes,
    n.date_autorisee_debut_transferts,
    n.date_autorisee_fin_transferts,
    code_d_r,
    code_ced
from
    refined_zone_gistrid.notifications_enriched n
where (siret_notifiant = :siret or siret_installation_traitement = :siret)
"""
