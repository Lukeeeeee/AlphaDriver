

def require_a_kwarg(name, kwargs):
    var = None
    for k, v in kwargs.items():
        if k is name:
            var = v
    if not var:
        raise Exception(("Missing a parameter '%s', call the method with %s=XXX" % (name, name)))
    else:
        return var

if __name__ == '__main__':
    t = {
        'a': 1
    }
    require_a_kwarg(name='a', kwargs=t)
