def parse_id(userid):
    return {'id': userid, 'hash': int(hash(userid))}

def parse_line(line):
    line = line.split()
    user = parse_id(line[0])
    song = parse_id(line[1])
    rating = float(line[2])

    return {'user': user, 'song':song, 'rating':rating}
