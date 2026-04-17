with source as (
    select * from {{ ref('raw_churn_labels') }}
)

select
    customer_id,
    case when lower(churn) = 'yes' then 1 else 0 end as is_churned
from source
