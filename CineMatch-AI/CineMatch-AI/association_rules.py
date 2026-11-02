"""
association_rules.py
--------------------
Association Rule Mining with the Apriori algorithm (mlxtend).

Each user is one "basket" containing the movies they liked (rated >= 4.0).
A rule such as   {Movie A} -> {Movie B}   means: users who liked A also tend to like B.

Key measures
    support     = share of all users who liked BOTH movies
    confidence  = of the users who liked A, the share who also liked B
    lift        = confidence / (share of users who liked B)
                  lift > 1 means A and B appear together MORE than by chance
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def generate_frequent_itemsets(basket, min_support=0.15, max_len=3):
    """Find itemsets (groups of movies) that appear in at least min_support of the baskets."""
    if basket is None or basket.empty:
        raise ValueError("The transaction table is empty.")
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be between 0 and 1.")
    itemsets = apriori(basket, min_support=min_support, max_len=max_len, use_colnames=True)
    itemsets["length"] = itemsets["itemsets"].apply(len)
    return itemsets.sort_values("support", ascending=False).reset_index(drop=True)


def generate_association_rules(itemsets, num_transactions, min_confidence=0.6, min_lift=1.0):
    """Create rules from frequent itemsets and keep the useful ones.

    Returns an empty DataFrame if no rule satisfies the thresholds.
    """
    if itemsets is None or itemsets.empty or (itemsets["length"] < 2).all():
        return pd.DataFrame()
    rules = association_rules(
        itemsets, num_itemsets=num_transactions, metric="confidence", min_threshold=min_confidence
    )
    rules = rules[rules["lift"] >= min_lift]
    return rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def _names(frozen, id_to_title):
    return " + ".join(id_to_title.get(i, str(i)) for i in sorted(frozen))


def itemsets_to_readable(itemsets, id_to_title):
    """Replace movie ids with titles so a human can read the table."""
    out = itemsets.copy()
    out["movies"] = out["itemsets"].apply(lambda s: _names(s, id_to_title))
    return out[["movies", "length", "support"]]


def rules_to_readable(rules, id_to_title):
    """Readable rules table: 'If user likes X  ->  also likes Y' with support/confidence/lift."""
    if rules is None or rules.empty:
        return pd.DataFrame(columns=["if_user_likes", "then_also_likes", "support", "confidence", "lift"])
    out = pd.DataFrame({
        "if_user_likes": rules["antecedents"].apply(lambda s: _names(s, id_to_title)),
        "then_also_likes": rules["consequents"].apply(lambda s: _names(s, id_to_title)),
        "support": rules["support"].round(3),
        "confidence": rules["confidence"].round(3),
        "lift": rules["lift"].round(3),
    })
    return out


def rules_for_movie(rules, movie_id, single_only=False):
    """Rules whose 'if' part contains the selected movie.

    single_only=True keeps only rules where the selected movie is the ONLY movie
    in the 'if' part (e.g. {A} -> {B}), which is the cleanest "A -> B" reading.
    """
    if rules is None or rules.empty:
        return pd.DataFrame()
    if single_only:
        mask = rules["antecedents"].apply(lambda s: s == frozenset([movie_id]))
    else:
        mask = rules["antecedents"].apply(lambda s: movie_id in s)
    return rules[mask]
