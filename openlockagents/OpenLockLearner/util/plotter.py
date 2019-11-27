import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple

MultiRunPlotData = namedtuple('MultiRunPlotData', ['name', 'data'])

fontsize = 13

def pad_uneven_python_list(v, fillval=0):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out


def smooth(y, box_pts):
    box_pts = max(box_pts, 1) if len(y) > box_pts else 1
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="valid")
    return y_smooth


def plot_information_gain_per_attempt(agent):
    create_plot(
          x_data=list(range(len(agent.information_gains_per_attempt))),
          y_data=agent.information_gains_per_attempt,
          x_label="Attempt Number",
          y_label="Information gain",
          title="Information gain per attempt",
          save_file=True,
          save_path=agent.writer.subject_path
          + "/Information_gain_per_attempt.pdf",
      )

def create_plot_from_multi_run_plot_data(multi_run_plot_datum, x_label, y_label, title, data_dir, save_fig=False, show_fig=True):
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    fig.clf()
    num_plots = 1
    clrs = sns.color_palette("husl", len(multi_run_plot_datum))
    base = fig.add_subplot(111)
    base.tick_params(labelcolor="w", top="off", bottom="off", left="off", right="off")
    with sns.axes_style("darkgrid"):
        axes = fig.subplots(1, num_plots)
        for ind in range(len(multi_run_plot_datum)):
            model_name = multi_run_plot_datum[ind].name
            model_data = multi_run_plot_datum[ind].data

            a_min = np.iinfo(np.int64).min
            a_max = np.iinfo(np.int64).max

            mean = np.mean(model_data, axis=0)
            std = np.std(model_data, axis=0)

            # when there is no need for extra smoothness
            # mean = np.mean(values, axis=0)
            # std = np.std(values, axis=0)

            axes.plot(
                np.arange(len(mean)), mean, "-", c=clrs[ind], label=model_name
            )
            axes.fill_between(
                np.arange(len(mean)),
                np.clip(mean - std, a_min=a_min, a_max=a_max),
                np.clip(mean + std, a_min=a_min, a_max=a_max),
                alpha=0.3,
                facecolor=clrs[ind],
            )

        axes.legend(fontsize=fontsize, loc=4)
    base.set_title(title, fontsize=fontsize)
    base.set_xlabel(x_label, fontsize=fontsize)
    base.set_ylabel(y_label, fontsize=fontsize)
    fig.tight_layout(pad=0.1)
    if show_fig:
        fig.show()
    if save_fig:
        save_path = data_dir + "/{}.pdf".format(title)
        print("Saving plot to {}".format(save_path))
        fig.savefig(save_path)


def create_plot(x_data, y_data, x_label, y_label, title, save_file=False, save_path=None, show_fig=False):
    if save_path is None:
        save_path = title

    fig = plt.figure()
    fig.set_size_inches(8, 5)
    fig.clf()
    with sns.axes_style("darkgrid"):
        sns.lineplot(
            x=x_data, y=y_data
        )

        # base.legend(fontsize=fontsize, loc=4)

    plt.ylabel(y_label, fontsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.tight_layout(pad=0)
    if show_fig:
        plt.show()

    if save_file:
        print("Saving plot to {}".format(save_path))
        fig.savefig(save_path)


def demo():

    repeat = 5
    length = 1000

    tag_list = ["Plot A", "Plot B", "Plot C"]

    value_list = [
        ("Model 1", np.random.rand(len(tag_list), repeat, length)),
        ("Model 2", np.random.rand(len(tag_list), repeat, length)),
        ("Model 3", np.random.rand(len(tag_list), repeat, length)),
        ("Model 4", np.random.rand(len(tag_list), repeat, length)),
    ]

    title = "Demo"

    fig = plt.figure()
    fig.set_size_inches(15, 5)
    fig.clf()
    num_plots = len(tag_list)
    clrs = sns.color_palette("husl", len(value_list))
    base = fig.add_subplot(111)
    base.spines["top"].set_color("none")
    base.spines["bottom"].set_color("none")
    base.spines["left"].set_color("none")
    base.spines["right"].set_color("none")
    base.tick_params(labelcolor="w", top="off", bottom="off", left="off", right="off")
    with sns.axes_style("darkgrid"):
        axes = fig.subplots(1, num_plots)
        for ind_model, item in enumerate(value_list):
            k, v = item
            values_list = v
            for ind, (tag, values) in enumerate(
                zip(np.array(tag_list), np.array(values_list))
            ):
                a_min = np.iinfo(np.int64).min
                a_max = np.iinfo(np.int64).max

                # when extra smoothness is needed for each repeated runnings
                tmp = []
                win_size = 10  # critic
                for i in values:
                    tmp.append(smooth(i, win_size))
                tmp = np.array(tmp)
                mean = np.mean(tmp, axis=0)
                std = np.std(tmp, axis=0)

                # when there is no need for extra smoothness
                # mean = np.mean(values, axis=0)
                # std = np.std(values, axis=0)

                axes[ind].plot(
                    np.arange(len(mean)), mean, "-", c=clrs[ind_model], label=k.upper()
                )
                axes[ind].fill_between(
                    np.arange(len(mean)),
                    np.clip(mean - std, a_min=a_min, a_max=a_max),
                    np.clip(mean + std, a_min=a_min, a_max=a_max),
                    alpha=0.3,
                    facecolor=clrs[ind_model],
                )
                axes[ind].set_ylabel(tag, fontsize=fontsize)

        axes[-1].legend(fontsize=fontsize, loc=4)  # inplot
        # axes[-1].legend(fontsize=fontsize, loc='center left', bbox_to_anchor=(1, 0.5))  # right side
        # axes[-1].legend(fontsize=fontsize, loc='center left', bbox_to_anchor=(0.5, -0.05))  # down side(not working yet)
    base.set_title(title, fontsize=fontsize)
    base.set_xlabel("Trials", fontsize=fontsize)
    fig.tight_layout(pad=0)
    fig.savefig("{}.pdf".format(title))


if __name__ == "__main__":
    demo()
