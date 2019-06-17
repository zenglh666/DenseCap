from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.fff
import models.fffv2
import models.fffv3
import models.fffv4
import models.fffv5
import models.fffv6
import models.fffv7
import models.fffv8
import models.fffv9
import models.fffva
import models.ttt
import models.rrr

def get_model(name, lrp=False):
    name = name.lower()

    if name == "fff":
        return models.fff.FFF
    elif name == "fffv2":
        return models.fffv2.FFF
    elif name == "fffv3":
        return models.fffv3.FFF
    elif name == "fffv4":
        return models.fffv4.FFF
    elif name == "fffv5":
        return models.fffv5.FFF
    elif name == "fffv6":
        return models.fffv6.FFF
    elif name == "fffv7":
        return models.fffv7.FFF
    elif name == "fffv8":
        return models.fffv8.FFF
    elif name == "fffv9":
        return models.fffv9.FFF
    elif name == "fffva":
        return models.fffva.FFF
    elif name == "ttt":
        return models.ttt.FFF
    elif name == "rrr":
        return models.rrr.FFF
    else:
        raise LookupError("Unknown model %s" % name)
