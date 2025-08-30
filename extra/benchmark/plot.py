import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    with open(sys.argv[1], 'r') as file:
        result = json.load(file)

    df = pd.DataFrame.from_records(
        result['benchmarks'],
        exclude=[
            'cpu_time',
            'iterations',
            'per_family_instance_index',
            'real_time',
            'repetition_index',
            'repetitions',
            'run_name',
            'run_type',
            'threads',
            'time_unit',
        ]
    )

    df = df[~df['name'].str.contains('intel')]
    df = df[~df['name'].str.contains('split')]
    df = df[~df['name'].str.contains('4_plan')]
    df = df[~df['name'].str.contains('q15')]
    # df = df[~df['name'].str.contains('simd')]
    # df = df[~df['name'].str.contains('v1')]

    for _, group in df.groupby(by='family_index'):
        name = group['name'].iloc[0].split('/')[0]
        # if name == 'fft_plan':
        #     continue
        # if name == 'split_fft_plan':
        #     continue
        group['size'] = group['name'].str.split('/').str[-1].astype(int)
        print(group)

        plt.semilogx(group['size'], group['items_per_second'], label=name)

    plt.xlabel('N')
    plt.ylabel('Items/s')
    plt.grid(which='both')
    plt.legend()
    plt.show()


main()
