import models.ppn

def get_model(name, lrp=False):
    name = name.lower()

    if name == "ppn":
        return models.ppn.Model
    else:
        raise LookupError("Unknown model %s" % name)
