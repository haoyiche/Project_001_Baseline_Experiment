import os
from PIL import Image


def audit_mvtec(root):

    report = {}


    for category in os.listdir(root):

        category_path = os.path.join(
            root,
            category
        )


        if not os.path.isdir(category_path):
            continue


        report[category] = {

            "train":0,

            "test":{},


        }


        train_path = os.path.join(
            category_path,
            "train"
        )


        if os.path.exists(train_path):

            for _,_,files in os.walk(train_path):

                for f in files:

                    if f.endswith(".png"):

                        report[category]["train"] += 1



        test_path = os.path.join(
            category_path,
            "test"
        )


        if os.path.exists(test_path):

            for defect in os.listdir(test_path):

                defect_path = os.path.join(
                    test_path,
                    defect
                )


                count = 0


                for _,_,files in os.walk(defect_path):

                    for f in files:

                        if f.endswith(".png"):

                            count += 1


                report[category]["test"][defect] = count


    return report



if __name__ == "__main__":


    root = (
        "data/raw/"
        "mvtec_anomaly_detection"
    )


    result = audit_mvtec(root)


    for category,data in result.items():

        print("================")
        print(category)

        print(
            "Train:",
            data["train"]
        )

        print(
            "Test:"
        )

        for k,v in data["test"].items():

            print(
                k,
                ":",
                v
            )