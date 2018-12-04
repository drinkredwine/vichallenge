from app.models import load_data
from app.models import train_model


class TestModels(object):

    def test_load(self):
        data = load_data('data/user_items.csv')
        assert len(data) == 6

    def test_train(self):
        data = load_data('data/user_items.csv')
        train_model(data)
