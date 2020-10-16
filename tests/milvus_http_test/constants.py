import utils

default_fields = utils.gen_default_fields()
default_binary_fields = utils.gen_default_fields(binary=True)

default_entity = utils.gen_entities(1)
default_raw_binary_vector, default_binary_entity = utils.gen_binary_entities(1)

default_entities = utils.gen_entities(utils.default_nb)
default_raw_binary_vectors, default_binary_entities = utils.gen_binary_entities(utils.default_nb)
