from Levenshtein import distance
import difflib


def similarity(s1, s2):
    normalized1 = s1.lower()
    normalized2 = s2.lower()
    matcher = difflib.SequenceMatcher(None, normalized1, normalized2)
    return matcher.ratio()


def checking_for_improvement(text_after, text_before, gs_name, st_info, name_file="IMAGE"):
    with open(gs_name, 'r', encoding="UTF-8") as f:
        wt_1 = 1 / (st_info['n'] + 1)
        wt_n = st_info['n'] * wt_1

        text_gs = f.read()
        similarity_before = similarity(text_gs, text_before)
        similarity_after = similarity(text_gs, text_after)

        st_info["total_similarity_before"] = wt_n * st_info["total_similarity_before"] + wt_1 * similarity_before
        st_info["total_similarity_after"] = wt_n * st_info["total_similarity_after"] + wt_1 * similarity_after

        size = len(text_gs)
        levenshtein_before = distance(text_gs, text_before) / size
        levenshtein_after = distance(text_gs, text_after) / size

        st_info["total_levenshtein_before"] = wt_n * st_info["total_levenshtein_before"] + wt_1 * levenshtein_before
        st_info["total_levenshtein_after"] = wt_n * st_info["total_levenshtein_after"] + wt_1 * levenshtein_after

        print()
        print(
            f"{name_file}:\tSimilarity: \t"
            f" Before: {100 * similarity_before :5.2f}%\t"
            f" After: {100 * similarity_after :5.2f}%")
        print(
            f"{name_file}:\tLevenshtein:\t"
            f" Before: {levenshtein_before:5.2f}\t"
            f" After: {levenshtein_after:5.2f}")
        print("=======================")

        st_info["n"] += 1
