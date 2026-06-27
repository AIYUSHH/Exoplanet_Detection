from catalog import *

df = download_catalog()

confirmed = get_confirmed_targets(df)
false_positive = get_false_positive_targets(df)
candidate = get_candidate_targets(df)

save_target_list(
    confirmed,
    "data/targets/confirmed.txt"
)

save_target_list(
    false_positive,
    "data/targets/false_positive.txt"
)

save_target_list(
    candidate,
    "data/targets/candidate.txt"
)