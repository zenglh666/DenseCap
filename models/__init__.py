import models.ppn
import models.ppnv2
import models.man

def get_model(name, lrp=False):
    name = name.lower()

    if name == "ppn":
        return models.ppn.Model
    if name == "ppnv2":
        return models.ppnv2.Model
    elif name == "man":
        return models.man.Model
    else:
        raise LookupError("Unknown model %s" % name)
