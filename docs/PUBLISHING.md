# Repository publishing

A live remote repository was not created because GitHub access was not enabled. The repository is complete locally and its real development history is retained in the ZIP and Git bundle. Do not claim it is hosted until you have pushed it and verified assessor access.

## Coventry GitHub

Create an empty private repository named `7043SCN-ChefsHat-Opponent-Modelling-RL` on your university GitHub server. Do not initialise another README if you intend to preserve the supplied local history. The ZIP omits the raw `.git` directory and instead includes `repository.bundle`, which preserves the actual committed history. After extracting, clone that portable bundle into a fresh working repository, then run these commands, replacing YOUR_USERNAME with your actual account name:

```bash
git clone repository.bundle publish-ready
cd publish-ready
git branch -M main
git remote add origin https://github.coventry.ac.uk/YOUR_USERNAME/7043SCN-ChefsHat-Opponent-Modelling-RL.git
git push -u origin main
```

Authenticate using the university’s supported Git credential method. Never put tokens in source files, notebooks, remote URLs, or screenshots. Grant assessors the required repository access and test the link using an assessor-accessible account. The brief recommends Coventry GitHub for Task 2.

## Personal GitHub alternative

Create an empty private repository in your own account, then use the same commands with `https://github.com/YOUR_USERNAME/7043SCN-ChefsHat-Opponent-Modelling-RL.git`. Confirm that using this host is acceptable and that assessors have access.

## Large checkpoint files

Each checkpoint includes replay and optimiser state for resumption. Check file sizes before pushing. GitHub rejects individual files above its file-size limit. Use Git LFS for checkpoints if required, or upload them as release assets and update reproduction instructions. The delivered ZIP includes them locally regardless of remote storage choice.

## Viva recording

Record your own 3–5 minute explanation and actual trained-agent demonstration using the provided deck and speaking script. Upload the recording to your approved university video platform, set assessor access, and add its real URL to README.md. No student viva recording or video link is fabricated. This remains a required student submission step.

No historical milestone commits are backdated. The supplied Git history reflects when this package was actually built; it cannot substitute for missed teaching-block milestones.
