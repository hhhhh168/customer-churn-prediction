with source as (
    select * from {{ ref('raw_customers') }}
)

select
    customer_id,
    lower(gender) as gender,
    senior_citizen,
    case when lower(partner) = 'yes' then 1 else 0 end as has_partner,
    case when lower(dependents) = 'yes' then 1 else 0 end as has_dependents,
    tenure as tenure_months
from source
