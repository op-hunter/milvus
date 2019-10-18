import pdb
import logging

logger = logging.getLogger("milvus_benchmark.parser")


def operations_parser(operations, run_type="performance"):
    definitions = operations[run_type]
    return definitions


def table_parser(table_name):
    tmp = table_name.split("_")
    # if len(tmp) != 5:
    #     return None
    data_type = tmp[0]
    table_size_unit = tmp[1][-1]
    table_size = tmp[1][0:-1]
    if table_size_unit == "m":
        table_size = int(table_size) * 1000000
    elif table_size_unit == "b":
        table_size = int(table_size) * 1000000000
    index_file_size = int(tmp[2])
    dimension = int(tmp[3])
    metric_type = str(tmp[4])
    return (data_type, table_size, index_file_size, dimension, metric_type)


def search_params_parser(param):
    # parse top-k, set default value if top-k not in param
    if "top_ks" not in param:
        top_ks = [10]
    else:
        top_ks = param["top_ks"]
    if isinstance(top_ks, int):
        top_ks = [top_ks]
    elif isinstance(top_ks, list):
        top_ks = list(top_ks)
    else:
        logger.warning("Invalid format top-ks: %s" % str(top_ks))

    # parse nqs, set default value if nq not in param
    if "nqs" not in param:
        nqs = [10]
    else:
        nqs = param["nqs"]
    if isinstance(nqs, int):
        nqs = [nqs]
    elif isinstance(nqs, list):
        nqs = list(nqs)
    else:
        logger.warning("Invalid format nqs: %s" % str(nqs))

    # parse nprobes
    if "nprobes" not in param:
        nprobes = [1]
    else:
        nprobes = param["nprobes"]
    if isinstance(nprobes, int):
        nprobes = [nprobes]
    elif isinstance(nprobes, list):
        nprobes = list(nprobes)
    else:
        logger.warning("Invalid format nprobes: %s" % str(nprobes))    

    return top_ks, nqs, nprobes