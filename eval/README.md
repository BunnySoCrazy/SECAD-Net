
# Model Evaluation Guide

To begin, you should prepare a text file that includes the names of the shapes within the subset of your dataset.
In our example, this is located at `./ABC/ABC_list.txt`.

Before you begin the actual evaluation process, it is essential to first fine-tune and test on the subset:

```bash
python fine-tuning.py -e ABC -g 0 -c best --subset <path_to_your_filename_list>
python test.py -e ABC -g 0 -c best --subset <path_to_your_filename_list>
```

After the initial steps, evaluate the results with the provided script:

```bash
cd eval
sh eval.sh
```

The results will be stored in `./ABC/eval_results` and `./ABC/eval_edge_results`.


> **Note:** This evaluation framework incorporates significant portions of code from two notable sources: [**BSP-Net**](https://github.com/czq142857/BSP-NET-original/blob/master/evaluation/edge_from_point.py) and [**UCSGNet**](https://github.com/kacperkan/ucsgnet/blob/c13b204361e59c5b92a7983e929305e17a906b65/ucsgnet/ucsgnet/evaluate_on_3d_data.py). We extend our sincere gratitude to the original authors for their contributions and emphasize the importance of crediting these resources when utilizing or modifying their code within your own evaluation workflows.
