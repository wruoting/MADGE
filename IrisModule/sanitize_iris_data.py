import plygdata as pg


def create_iris_data(validation_data_ratio):
    iris_path = './SampleData/IrisData/iris.data'
    iris_data = []
    with open(iris_path) as file:
        for entry in file.readlines():
            # TODO: need to make entry[-1]
            iris_data.append(entry)
    return pg.split_data(iris_data, validation_size=validation_data_ratio)