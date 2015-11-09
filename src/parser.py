def parse_id(userid):

    # from some reason I have to use 7 'f's instead of 8 to get the
    # right number of bits or else spark will complain about not
    # being able to cast from a long to an integer.
    return {'id': userid, 'hash': int(hash(userid) & 0xfffffff)}

def parse_line(line):
    line = line.split()
    user = parse_id(line[0])
    song = parse_id(line[1])
    rating = float(line[2])

    return {'user':user, 'song':song, 'rating':rating}
