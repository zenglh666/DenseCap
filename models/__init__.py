import models.ppn
import models.man

def get_model(name, lrp=False):
    name = name.lower()

    if name == "ppn":
        return models.ppn.Model
    elif name == "man":
        return models.man.Model
    else:
        raise LookupError("Unknown model %s" % name)
