"""GIN-IPA NaN watchdog scanner (runs on any cluster, stdlib only). Args: RUN_ID-glob root... (repo benchmark/02_tasks, or a TamIA scratch root)
Prints one TSV line per fold training log: run_dir_basename fold log last_epoch first_nonfinite_epoch n_nonfinite_epochs last_dice_mean"""
import glob, os, re, sys
rid = sys.argv[1]
for root in sys.argv[2:]:
    for log in sorted(glob.glob(os.path.join(root, '*', '8_results*', '01_predictions', '*', '*', 'auglab', rid, 'Dataset*', '*', 'fold_*', 'training_log_*.txt'))
                      + glob.glob(os.path.join(root, '*', '*', '8_results*', '01_predictions', '*', '*', 'auglab', rid, 'Dataset*', '*', 'fold_*', 'training_log_*.txt'))):
        ep, first, nbad, dice = -1, '', 0, ''
        try:
            for line in open(log, errors='replace'):
                m = re.search(r': Epoch (\d+)\s*$', line)
                if m:
                    ep = int(m.group(1)); continue
                m = re.search(r'(train_loss|val_loss|train loss_seg|loss_consist) (\S+)', line)
                if m and re.match(r'-?(nan|inf)', m.group(2).rstrip(','), re.I):
                    if first == '': first = ep
                    if ep != getattr(sys, '_lastbad', None): nbad += 1; sys._lastbad = ep
                m = re.search(r'Pseudo dice \[(.*)\]', line)
                if m:
                    v = [float(x) for x in re.findall(r'([-\d.]+(?:e-?\d+)?|nan)\)?(?:,|$)', m.group(1).replace('np.float32(', ''))]
                    dice = '%.3f' % (sum(v) / len(v)) if v else ''
        except OSError:
            continue
        fold = re.search(r'/fold_(\d)/', log).group(1)
        rd = log.split('/auglab/')[1].split('/')[0]
        print('\t'.join(str(x) if str(x) != '' else '-' for x in [rd, fold, log, ep, first, nbad, dice]))  # '-' = empty: bash read collapses empty tab fields
