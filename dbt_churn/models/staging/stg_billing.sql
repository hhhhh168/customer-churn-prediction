with source as (
    select * from {{ ref('raw_billing') }}
)

select
    customer_id,
    lower(contract) as contract_type,
    case when lower(paperless_billing) = 'yes' then 1 else 0 end as is_paperless,
    lower(payment_method) as payment_method,
    monthly_charges,
    case
        when total_charges = '' or total_charges is null then 0.0
        else cast(total_charges as double)
    end as total_charges
from source
