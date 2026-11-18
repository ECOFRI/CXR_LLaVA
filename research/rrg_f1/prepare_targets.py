"""Prepare the ignored Findings cache before the timed 200-report labeler gate."""
from research.rrg_f1.labeler import train_targets

if __name__ == '__main__':
    rows, targets = train_targets()
    print('FINDINGS_CACHE_READY rows=' + str(len(rows)), flush=True)
