with customers as (
    select * from {{ ref('stg_customers') }}
),

services as (
    select * from {{ ref('stg_services') }}
),

billing as (
    select * from {{ ref('stg_billing') }}
),

churn as (
    select * from {{ ref('stg_churn_labels') }}
),

addon_counts as (
    select
        customer_id,
        (  (case when online_security  = 'yes' then 1 else 0 end)
         + (case when online_backup    = 'yes' then 1 else 0 end)
         + (case when device_protection = 'yes' then 1 else 0 end)
         + (case when tech_support     = 'yes' then 1 else 0 end)
         + (case when streaming_tv     = 'yes' then 1 else 0 end)
         + (case when streaming_movies = 'yes' then 1 else 0 end)
        ) as addon_count
    from services
)

select
    c.customer_id,
    c.gender,
    c.senior_citizen,
    c.has_partner,
    c.has_dependents,
    c.tenure_months,
    case
        when c.tenure_months <= 6  then '0-6 mo'
        when c.tenure_months <= 12 then '7-12 mo'
        when c.tenure_months <= 24 then '13-24 mo'
        when c.tenure_months <= 48 then '25-48 mo'
        else '49+ mo'
    end as tenure_bucket,
    s.phone_service,
    s.multiple_lines,
    s.internet_service,
    s.online_security,
    s.online_backup,
    s.device_protection,
    s.tech_support,
    s.streaming_tv,
    s.streaming_movies,
    a.addon_count,
    b.contract_type,
    b.is_paperless,
    b.payment_method,
    b.monthly_charges,
    b.total_charges,
    b.monthly_charges * 12 as annual_revenue,
    ch.is_churned
from customers c
left join services s on c.customer_id = s.customer_id
left join billing b on c.customer_id = b.customer_id
left join addon_counts a on c.customer_id = a.customer_id
left join churn ch on c.customer_id = ch.customer_id
