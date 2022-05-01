import os

class Dialog:
    def __init__(self):
        self.phrase_pairs = []

    def convert_to_int_list(self, arr):
        return list(map(lambda x: int(x), arr))

    def process_daily_dialogs(self, emo_path, diag_path):
        end_phrase_token = "__eou__"

        with open(emo_path) as f:
            emos = f.readlines()
            emos = list(map(lambda x: self.convert_to_int_list(x.split(" ")[:-1]), emos))  # drop last "\n"
            # emos = list(map(lambda x: int(x), emos))
            print(emos)

        prompts = []
        answers = []

        with open(diag_path) as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                diag_list = line.split(end_phrase_token)[:-1]  # drop last "\n"

                if len(diag_list) % 2 != 0:
                    diag_list = diag_list[:-1]

                emos_item = emos[index]

                # print(len(diag_list[:-1]))
                for j, item in enumerate(diag_list):

                    if j % 2 == 0:
                        # print("HERE")
                        prompts.append(item)
                    else:
                        # print("HERE2")
                        answers.append(item)

                    # print(item, emo_dict[emos_item[j]], j%2==0)

                # print("\n \n \n")
if __name__ == "__main__":
