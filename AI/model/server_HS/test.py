from model_load_HS import HandSignModel
from mock import test_data, test_data_2


# Handle the webapp
def handle_coordinate(data):
    # print('coordinate', data)
    model = HandSignModel(mode='A')
    result = model.predict(data)
    return result


if __name__ == '__main__':
    '''
    test_data 는 'a' 데이터
    test_data_2 는 'angel' 데이터 입니다.
    '''
    print("result: ", handle_coordinate(data=test_data))
