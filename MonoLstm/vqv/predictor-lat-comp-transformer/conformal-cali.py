import numpy as np
from netcal.scaling import TemperatureScaling
from netcal.metrics import ECE
from netcal.binning import HistogramBinning, IsotonicRegression, ENIR, BBQ
from netcal.scaling import LogisticCalibration, TemperatureScaling, BetaCalibration
from netcal.metrics import ACE, ECE, MCE
from netcal.presentation import ReliabilityDiagram
import matplotlib.pyplot as plt
import transformer
import pandas as pd


def main(step, testpath, validpath):
    data = {}
    data["sft"] = {}
    data["lbl"] = {}
    data_vali = {}
    data_vali["sft"] = {}
    data_vali["lbl"] = {}
    # Shift labels horizon steps away.
    (
        data["sft"][step],
        data["lbl"][step],
        data_vali["sft"][step],
        data_vali["lbl"][step],
    ) = transformer.eval_train_cc(
        csv_path="../safety_detection_labeled_data/Safety_Detection_Labeled.csv",
        images_folder="../safety_detection_labeled_data/",
        vae_weights="./weights/vae_weights_split.pth",
        transformer_weights=f"./weights/transformer_weights{step}.pth",
        seq_len=32,
        horizon=step,
        load_transformer_weights=True,
        load_d=True,
        data=None,
        device="cpu",
    )

    print(data["sft"][step].shape)
    print(data["lbl"][step].shape)
    print(data_vali["sft"][step].shape)
    print(data_vali["lbl"][step].shape)

    # Save training data to CSV
    df_train = pd.DataFrame()
    df_train["softmax"] = data["sft"][step]
    df_train["label"] = data["lbl"][step]
    df_train.to_csv(
        f"./reliability_results/output_data/train_step{step}.csv", index=False
    )

    # Save validation data to CSV
    df_vali = pd.DataFrame()
    df_vali["softmax"] = data_vali["sft"][step]
    df_vali["label"] = data_vali["lbl"][step]
    df_vali.to_csv(f"./reliability_results/output_data/val_step{step}.csv", index=False)

    n_bins = 10
    bins = 10
    hist_bins = 20

    ece = ECE(n_bins)
    # data_vali = np.load(testpath)
    # data = np.load(validpath)
    totalsft = data["lbl"]
    build_set_sm = data["sft"][step]
    build_set_gt = data["lbl"][step]
    sft_vali = data_vali["sft"][step]
    lbl_vali = data_vali["lbl"][step]
    confidences = sft_vali
    ground_truth = lbl_vali
    temperature = TemperatureScaling()

    histogram = HistogramBinning(hist_bins)
    iso = IsotonicRegression()
    bbq = BBQ()
    enir = ENIR()
    method = "mle"

    lr_calibration = LogisticCalibration(detection=False, method=method)
    temperature = TemperatureScaling(detection=False, method=method)
    betacal = BetaCalibration(detection=False, method=method)

    models = [
        ("hist", histogram),
        ("iso", iso),
        ("bbq", bbq),
        # ("enir", enir),
        ("lr", lr_calibration),
        ("temperature", temperature),
        ("beta", betacal),
    ]

    ace = ACE(bins)
    ece = ECE(bins)
    mce = MCE(bins)
    validation_set_sm = confidences
    validation_set_gt = ground_truth
    predictions = []

    ece_precal = [ece.measure(validation_set_sm, validation_set_gt)]
    method_num = np.argmin(ece_precal)
    print(f"ECE: {ece_precal[method_num]}")

    with open("./reliability_results/ece_scores.txt", "a") as file:
        file.write(f"Precalibrated ECE: {ece_precal[method_num]}\n")

    diagram = ReliabilityDiagram(bins=bins, title_suffix="default")
    diagram.plot(
        validation_set_sm,
        validation_set_gt,
        filename="./reliability_results/"
        + str(step)
        + "precal"
        + str(ece_precal[method_num])
        + ".png",
    )

    all_ace = []
    all_ece = []
    all_mce = []
    for model in models:
        name, instance = model
        print("Build %s model" % name)
        instance.fit(build_set_sm, build_set_gt)

    for model in models:
        _, instance = model
        prediction = instance.transform(validation_set_sm)
        predictions.append(prediction)

        all_ace.append(ace.measure(prediction, validation_set_gt))
        all_ece.append(ece.measure(prediction, validation_set_gt))
        all_mce.append(mce.measure(prediction, validation_set_gt))
    x = [0, 1]
    # plt.plot(x, all_ece, x, all_mce)
    # plt.plot(
    #     x,
    #     all_ece,
    # )
    # plt.show()
    print(all_ece)
    print(all_mce)
    bins2 = np.linspace(0.1, 1, bins)

    # diagram = ReliabilityDiagram(bins=bins, title_suffix="default")
    # diagram.plot(
    #     validation_set_sm,
    #     validation_set_gt,
    #     filename="./" + str(step) + "test" + str(all_ece[0]) + ".png",
    # )

    method_num = np.argmin(all_ece)
    print(f"ECE: {all_ece[method_num]}")

    with open("./reliability_results/ece_scores.txt", "a") as file:
        file.write(f"ECE: {all_ece[method_num]}\n")
        file.write("\n")
    diagram = ReliabilityDiagram(bins=bins, title_suffix=models[method_num])
    prediction = predictions[method_num]
    diagram.plot(
        prediction,
        validation_set_gt,
        filename="./reliability_results/" + str(step) + ".png",
        title_suffix="",
    )

    binned = np.digitize(prediction, bins2)
    dset = []
    binsamount = []
    for i in range(10):
        posi = list(np.where(binned == i))[0]
        binsamount.append(len(posi))
        if len(posi) > 30:
            dsubset = []
            for im in range(200):
                temp_cali = []
                temp_gt = []
                for jm in range(1000):
                    inumber = np.random.randint(0, len(posi))
                    temp_cali.append(prediction[inumber])
                    temp_gt.append(validation_set_gt[inumber])
                temp_cali = np.array(temp_cali)
                temp_gt = np.array(temp_gt)
                mu_cali = np.mean(temp_cali)
                mu_gt = np.mean(temp_gt)
                dsubset.append(abs(mu_cali - mu_gt))
            dsubset_np = np.array(dsubset)
            dsubset_np = np.sort(dsubset_np)
            dset.append(dsubset_np)
        else:
            dset.append(0)

            # ki = (1-0.9)*(1+ni/2)
    # print(binsamount)
    # np.savez_compressed(
    #     str(step) + "k95200.npz",
    #     dset=dset,
    #     lbl=validation_set_gt,
    #     cali=prediction,
    #     ori=validation_set_sm,
    #     ece=all_ece,
    #     mce=all_mce,
    #     number=method_num,
    # )
    print("end")
    return all_ece[0], all_ece[method_num], all_ace[0], all_ace[method_num]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="conformal calibration")
    parser.add_argument(
        "--valid",
    )
    parser.add_argument(
        "--test",
    )

    args = parser.parse_args()

    valid_path = args.valid
    test_path = args.test
    testpath = test_path
    validpath = valid_path
    ece = []
    ece2 = []
    mce = []
    mce2 = []
    for i in range(10, 101, 10):
        print(i)
        a, b, c, d = main(i, testpath, validpath)
        ece2.append(b)
        mce2.append(d)
        ece.append(a)
        ece.append(c)
        print(ece)
        print(ece2)
        print(mce)
        print(mce2)
