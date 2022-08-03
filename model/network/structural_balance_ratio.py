# coding utf-8

import os
import csv
import numpy as np


DATAPATH = os.path.join(os.getcwd(), 'generated', 'trial')
model_dir = os.path.join(DATAPATH)
csv_file = 'BN_n_150_p_0.05.txt.csv'

f = open(os.path.join(model_dir, csv_file), 'r', encoding='utf-8')

motif = np.zeros([132, 6])
i = 0
k = 0

with open(os.path.join(model_dir, csv_file)) as f:
    reader = csv.reader(f)
    for row in reader:
        i += 1
        if i > 32:
            if (i - 32) % 4 == 1:
                motif[k, 0] = row[0]
                motif[k, 1] = row[1]
                motif[k, 4] = row[2][:-1]
                motif[k, 5] = row[3][:-1]
            elif (i - 32) % 4 == 2:
                motif[k, 2] = row[1]
            elif (i - 32) % 4 == 3:
                motif[k, 3] = row[1]
            elif (i - 32) % 4 == 0:
                k += 1


M38_CB = [0, 0]
M38_PB = [0, 0]
M38_CI = [0, 0]
for j in range(len(motif)):
    if motif[j, 0] == 38:
        if motif[j, 2] == 100:
            if motif[j, 3] == 220:
                M38_CB[0] += motif[j, 4]
                M38_CB[1] += motif[j, 5]
            elif motif[j, 3] == 110:
                M38_CB[0] += motif[j, 4]
                M38_CB[1] += motif[j, 5]
            elif motif[j, 3] == 120:
                M38_CI[0] += motif[j, 4]
                M38_CI[1] += motif[j, 5]
            elif motif[j, 3] == 210:
                M38_CI[0] += motif[j, 4]
                M38_CI[1] += motif[j, 5]
        elif motif[j, 2] == 200:
            if motif[j, 3] == 210:
                M38_CB[0] += motif[j, 4]
                M38_CB[1] += motif[j, 5]
            elif motif[j, 3] == 120:
                M38_CB[0] += motif[j, 4]
                M38_CB[1] += motif[j, 5]
            elif motif[j, 3] == 110:
                M38_CI[0] += motif[j, 4]
                M38_CI[1] += motif[j, 5]
            elif motif[j, 3] == 220:
                M38_CI[0] += motif[j, 4]
                M38_CI[1] += motif[j, 5]

if M38_CB[0] + M38_PB[0] + M38_CI[0] == 0:
    BR_38 = 0
else:
    BR_38 = (M38_CB[0] + 0.5 * M38_PB[0]) / (M38_CB[0] + M38_PB[0] + M38_CI[0])
if M38_CB[1] + M38_PB[1] + M38_CI[1] == 0:
    BR_38_random = 0
else:
    BR_38_random = (M38_CB[1] + 0.5 * M38_PB[1]) / (M38_CB[1] + M38_PB[1] + M38_CI[1])

M46_CB = [0, 0]
M46_PB = [0, 0]
M46_CI = [0, 0]

for j in range(len(motif)):
    if motif[j, 0] == 46:
        if motif[j, 2] == 101:
            if motif[j, 3] == 110:
                M46_CB[0] += motif[j, 4]
                M46_CB[1] += motif[j, 5]
            else:
                M46_PB[0] += motif[j, 4]
                M46_PB[1] += motif[j, 5]
        elif motif[j, 2] == 102:
            if motif[j, 3] == 120:
                M46_CI[0] += motif[j, 4]
                M46_CI[1] += motif[j, 5]
            elif motif[j, 3] == 210:
                M46_PB[0] += motif[j, 4]
                M46_PB[1] += motif[j, 5]
            elif motif[j, 3] == 220:
                M46_CB[0] += motif[j, 4]
                M46_CB[1] += motif[j, 5]
        elif motif[j, 2] == 201:
            if motif[j, 3] == 110:
                M46_CI[0] += motif[j, 4]
                M46_CI[1] += motif[j, 5]
            elif motif[j, 3] == 210:
                M46_CB[0] += motif[j, 4]
                M46_CB[1] += motif[j, 5]
            elif motif[j, 3] == 220:
                M46_PB[0] += motif[j, 4]
                M46_PB[1] += motif[j, 5]
        else:
            M46_CI[0] += motif[j, 4]
            M46_CI[1] += motif[j, 5]
if M46_CB[0] + M46_PB[0] + M46_CI[0] == 0:
    BR_46 = 0
    BR_46_random = 0
else:
    BR_46 = (M46_CB[0] + 0.5 * M46_PB[0]) / (M46_CB[0] + M46_PB[0] + M46_CI[0])
    if M46_CB[1] + M46_PB[1] + M46_CI[1] == 0:
        BR_46_random = 0
    else:
        BR_46_random = (M46_CB[1] + 0.5 * M46_PB[1]) / (M46_CB[1] + M46_PB[1] + M46_CI[1])

if M46_CB[0] + M46_PB[0] + M46_CI[0] == 0:
    BR_46 = 0
else:
    BR_46 = (M46_CB[0] + 0.5 * M46_PB[0]) / (M46_CB[0] + M46_PB[0] + M46_CI[0])
if M46_CB[1] + M46_PB[1] + M46_CI[1] == 0:
    BR_46_random = 0
else:
    BR_46_random = (M46_CB[1] + 0.5 * M46_PB[1]) / (M46_CB[1] + M46_PB[1] + M46_CI[1])

M166_CB = [0, 0]
M166_PB = [0, 0]
M166_CI = [0, 0]
for j in range(len(motif)):
    if motif[j, 0] == 166:
        if motif[j, 1] == 10:
            if motif[j, 2] == 100:
                if motif[j, 3] == 210:
                    M166_CI[0] += motif[j, 4]
                    M166_CI[1] += motif[j, 5]
                else:
                    M166_CB[0] += motif[j, 4]
                    M166_CB[1] += motif[j, 5]
            else:
                M166_PB[0] += motif[j, 4]
                M166_PB[1] += motif[j, 5]
        else:
            if motif[j, 3] == 210:
                M166_CB[0] += motif[j, 4]
                M166_CB[1] += motif[j, 5]
            else:
                M166_CI[0] += motif[j, 4]
                M166_CI[1] += motif[j, 5]
if M166_CB[0] + M166_PB[0] + M166_CI[0] == 0:
    BR_166 = 0
else:
    BR_166 = (M166_CB[0] + 0.5 * M166_PB[0]) / (M166_CB[0] + M166_PB[0] + M166_CI[0])
if M166_CB[1] + M166_PB[1] + M166_CI[1] == 0:
    BR_166_random = 0
else:
    BR_166_random = (M166_CB[1] + 0.5 * M166_PB[1]) / (M166_CB[1] + M166_PB[1] + M166_CI[1])

M238_CB = [0, 0]
M238_PB = [0, 0]
M238_CI = [0, 0]
for j in range(len(motif)):
    if motif[j, 0] == 238:
        if (motif[j, 1] == 11 and motif[j, 2] == 101 and motif[j, 3] == 110):
            M238_CB[0] += motif[j, 4]
            M238_CB[1] += motif[j, 5]
        elif (motif[j, 1] == 11 and motif[j, 2] == 102 and motif[j, 3] == 120):
            M238_CI[0] += motif[j, 4]
            M238_CI[1] += motif[j, 5]
        elif (motif[j, 1] == 12 and motif[j, 2] == 102 and motif[j, 3] == 220):
            M238_CB[0] += motif[j, 4]
            M238_CB[1] += motif[j, 5]
        elif (motif[j, 1] == 22 and motif[j, 2] == 202 and motif[j, 3] == 220):
            M238_CI[0] += motif[j, 4]
            M238_CI[1] += motif[j, 5]
        elif ((motif[j, 1] == 11 and motif[j, 2] == 101 and motif[j, 3] == 220) or
              (motif[j, 1] == 11 and motif[j, 2] == 201 and motif[j, 3] == 210) or
              (motif[j, 1] == 12 and motif[j, 2] == 201 and motif[j, 3] == 220)):
            M238_CB[0] += 2 / 3 * motif[j, 4]
            M238_CI[0] += 1 / 3 * motif[j, 4]
            M238_CB[1] += 2 / 3 * motif[j, 5]
            M238_CI[1] += 1 / 3 * motif[j, 5]
        elif ((motif[j, 1] == 11 and motif[j, 2] == 201 and motif[j, 3] == 120) or
              (motif[j, 1] == 11 and motif[j, 2] == 202 and motif[j, 3] == 220) or
              (motif[j, 1] == 21 and motif[j, 2] == 201 and motif[j, 3] == 220)):
            M238_CB[0] += 1 / 3 * motif[j, 4]
            M238_CI[0] += 2 / 3 * motif[j, 4]
            M238_CB[1] += 1 / 3 * motif[j, 5]
            M238_CI[1] += 2 / 3 * motif[j, 5]
        else:
            M238_PB[0] += motif[j, 4]
            M238_PB[1] += motif[j, 5]
if M238_CB[0] + M238_PB[0] + M238_CI[0] == 0:
    BR_238 = 0
else:
    BR_238 = (M238_CB[0] + 0.5 * M238_PB[0]) / (M238_CB[0] + M238_PB[0] + M238_CI[0])
if M238_CB[1] + M238_PB[1] + M238_CI[1] == 0:
    BR_238_random = 0
else:
    BR_238_random = (M238_CB[1] + 0.5 * M238_PB[1]) / (M238_CB[1] + M238_PB[1] + M238_CI[1])

Balance_Ratio = 0
k = 0
for i in [BR_38, BR_46, BR_166, BR_238]:
    if i != 0:
        Balance_Ratio += i
        k += 1
Balance_Ratio = Balance_Ratio / k
Balance_Ratio_Random = 0
k = 0
for j in [BR_38_random, BR_46_random, BR_166_random, BR_238_random]:
    if j != 0:
        Balance_Ratio_Random += j
        k += 1
Balance_Ratio_Random = Balance_Ratio_Random / k

print("Global balance ratio is:", Balance_Ratio)