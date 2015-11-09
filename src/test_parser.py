from src.parser import parse_id, parse_line


def test_parse_userid():
    userids =  ['00020fcd8b01986a6a85b896ccde6c49f35142ad', '0003477fcf455dc4fcae3d7ca5e329cef811c868']
    for userid in userids:
        userid_dict = parse_id(userid)
        assert isinstance(userid_dict['id'], str)
        assert isinstance(userid_dict['hash'], int)
        print (userid_dict)

def test_parse_line():
    line = '00020fcd8b01986a6a85b896ccde6c49f35142ad	SOPKEIV12AB018220D	1'
    parsed_line = parse_line(line)
    assert 3 == len(parsed_line)
    print parsed_line

test_parse_userid()