def check_if_composite(options, electrode):
    if options["particle phases"] == "2":
        return True
    elif (
        isinstance(options["particle phases"], tuple)
        and options["particle phases"][0] == "2"
    ):
        if electrode == "positive":
            return False
        else:
            return True
    elif (
        isinstance(options["particle phases"], tuple)
        and options["particle phases"][1] == "2"
    ):
        if electrode == "positive":
            return True
        else:
            return False
    else:
        return False
