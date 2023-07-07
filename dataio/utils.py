from os.path import exists

def check_metadataset_ready(problem_type="regression", features="efficient", path=None):
    if path is None:
        path = f"./assets/{problem_type}/metadatasets"

    return exists(f"{path}/{features}.csv")
    