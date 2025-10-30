import argparse
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path', type=str, required=True,
                        help='path to the data to be drawn')
    parser.add_argument('-s', '--start_epoch', type=int, required=True,
                        help='draw from which starting epoch')
    parser.add_argument('-i', '--interval_epoch', type=int, required=True,
                        help='interval between each epoch drawn')
    parser.add_argument('-t', '--title', type=str, required=True,
                        help='title of the graph')
    parser.add_argument('-x', '--x_label', type=str, required=True,
                        help='x_axis of the graph')
    parser.add_argument('-y', '--y_label', type=str, required=True,
                        help='y_axis of the graph')
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='output path of the graph')
    args = parser.parse_args()

    data_path = args.data_path
    start_epoch = args.start_epoch
    interval_epoch = args.interval_epoch
    x_label = args.x_label
    y_label = args.y_label
    title = args.title
    out_path = args.out_path

    numbers_list = []
    with open(data_path, "r") as file:
        for line in file:
            number = float(line.strip())  #Convert the line to a number (assuming it's a floating-point number)
            numbers_list.append(number)

    x_axis = [start_epoch + (interval_epoch * x) for x in range(len(numbers_list))]
    plt.plot(x_axis, numbers_list)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(out_path)
