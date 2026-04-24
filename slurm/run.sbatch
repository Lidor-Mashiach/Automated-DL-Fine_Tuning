#!/bin/bash
# ==============================================================================
#  AutoTune-NN | SLURM submission script
# ==============================================================================
#
#  הגשה:   sbatch slurm/run.sbatch
#  הצגת סטטוס:   squeue -u $USER
#  ביטול:   scancel <jobid>
#
#  ערוך את הפרמטרים למטה לפי האשכול הספציפי שלך.
#  במיוחד: --partition (שם תור), --gres (מספר GPUs), --time (מגבלת זמן).
# ==============================================================================

#SBATCH --job-name=autotune_nn
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --time=08:00:00                  # HH:MM:SS - מגבלת זמן קשיחה של SLURM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                # CPUs ל-data loading workers
#SBATCH --mem=16G                        # RAM
#SBATCH --partition=main                 # ערוך לשם התור הנכון באשכול שלך
#SBATCH --gres=gpu:1                     # GPU יחיד; לשנות ל-gpu:2 וכו' אם צריך

# יצירת תיקיית לוגים אם לא קיימת
mkdir -p slurm_logs

# ---- סביבה ----
# אם יש module system באשכול:
# module load python/3.11
# module load cuda/12.1

# הפעלת venv/conda (שנה לפי ההגדרות שלך):
# source ~/venvs/autotune_nn/bin/activate
# או:  conda activate autotune_nn

# מעבר לתיקיית הפרויקט
cd "$SLURM_SUBMIT_DIR" || exit 1

# הדפסת מידע סביבתי ללוג (עוזר בדיבאג בדיעבד)
echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Python:    $(which python)"
echo "Working:   $(pwd)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "  (no nvidia-smi)"
echo "========================================"

# ---- הרצה בפועל ----
python main.py

echo "========================================"
echo "Job finished at $(date)"
echo "========================================"
