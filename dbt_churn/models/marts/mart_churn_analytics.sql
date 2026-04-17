with features as (
    select * from {{ ref('int_customer_features') }}
)

select
    *,

    case
        when addon_count = 0 then '0'
        when addon_count <= 2 then '1-2'
        when addon_count <= 4 then '3-4'
        else '5-6'
    end as addon_bucket,

    case
        when contract_type = 'month-to-month'
             and tenure_months <= 6
             and internet_service = 'fiber optic'
             and addon_count = 0
            then 'critical'
        when contract_type = 'month-to-month'
             and tenure_months <= 12
            then 'high'
        when contract_type = 'month-to-month'
            then 'medium'
        else 'low'
    end as risk_segment

from features
