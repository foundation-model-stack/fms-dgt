# Local
from fms_dgt.blocks.validators.rouge import RougeDedupValidator


class TestRougeValidator:
    def test_seq(self):
        thresh = 0.91
        all_data = [
            "I went to the store yesterday",
            "blue red yellow green",
        ]
        inputx = "blue red yellow green one two three four five six seven eight nine ten black blue orange white".split(
            " "
        )
        # this is sorted into the order rouge will use,
        # since the longer this string is, the larger its rouge score
        inputsx = [" ".join(inputx[:n]) for n in reversed(range(6, len(inputx)))]
        validator = RougeDedupValidator(name="test_rouge_validator", threshold=thresh)

        # When we add one at a time, five pass the test,
        # since every third item or so is far enough from its neighbors to pass.
        inps1 = []
        for inpx in inputsx:
            inp1 = [{"a": inpx}]
            print("INPS", all_data + inps1)
            validator.generate(
                inp1, context=all_data + inps1, arg_fields=["a"], result_field="result"
            )
            print(f'add 1 resx:   {inp1[0]["result"]:1}      {inp1[0]["a"]}')
            if inp1[0]["result"]:
                inps1.append(inpx)

        # however, it we add all at once, only the first is added,
        # since you use all the inputs in the filtering,
        # even if they don't pass.
        # each is too close to its neighbor
        inputs = [{"a": x} for x in inputsx]

        validator = RougeDedupValidator(name="test_rouge_validator", threshold=thresh)
        validator.generate(
            inputs, context=all_data, arg_fields=["a"], result_field="result"
        )
        for inp in inputs:
            print(f'add all res:  {inp["result"]:1}      {inp["a"]}')

        # # add them one at a time
        # validator = RougeDedupValidator(name="test_rouge_validator", threshold=thresh)
        # for inp in inputs:
        #     validator.generate(
        #         [inp], context=all_data, arg_fields=["a"], result_field="result"
        #     )
        #     print("res", inp["result"])
