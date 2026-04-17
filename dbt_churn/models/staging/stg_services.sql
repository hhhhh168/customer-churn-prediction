with source as (
    select * from {{ ref('raw_services') }}
)

select
    customer_id,
    lower(phone_service) as phone_service,
    lower(multiple_lines) as multiple_lines,
    lower(internet_service) as internet_service,
    lower(online_security) as online_security,
    lower(online_backup) as online_backup,
    lower(device_protection) as device_protection,
    lower(tech_support) as tech_support,
    lower(streaming_tv) as streaming_tv,
    lower(streaming_movies) as streaming_movies
from source
