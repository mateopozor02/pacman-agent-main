# Previous-edition teams

`my_team.py` of teams from earlier editions of the course, shared by the instructor.
Files are unmodified copies (some use old-style imports such as `from game import ...`;
`scripts/eval.py` handles those). Use them as opponents:

```bash
python scripts/eval.py mateo --pool prev                 # every team in this folder
python scripts/eval.py mateo --pool prev/foxtrotdelta_hub prev/supergodgg
```

| file | source | commit |
|------|--------|--------|
| `aiko_vdm.py` | https://github.com/Aiko-vdm/DotGobblers | 3384ece |
| `amadeohm.py` | https://github.com/amadeohm/AutSys-pacman | 500bd63 |
| `cherryneosea.py` | https://github.com/cherryneosea/TheDotDominators | 3d07590 |
| `dea2001.py` | https://github.com/dea2001/pacManContest | 3da1dfd |
| `foxtrotdelta_hub.py` | https://github.com/foxtrotdelta-hub/flagrats | a622e16 |
| `lorecleme.py` | https://github.com/lorecleme/pacman-agent | 880db95 |
| `maferfarfan2122.py` | https://github.com/maferfarfan2122/pacman_AS | a2fd2e7 |
| `marinadalmau.py` | https://github.com/marinadalmau/pacman-agent | 2fd027b |
| `rcrtss.py` | https://github.com/rcrtss/pacman-agent | ff7b88c |
| `robpriem.py` | https://github.com/robpriem/PacBoys | 67d419d |
| `scooooooott.py` | https://github.com/Scooooooott/Pacman_IDontThinkWeCanFinishThisToday | af498c1 |
| `sebastiantabacariu.py` | https://github.com/SebastianTabacariu/TooPac | c6cbae9 |
| `sta05.py` | https://github.com/sta05/Clingho_pacman | fb19830 |
| `supergodgg.py` | https://github.com/Supergodgg/LL_Autonomous_Pacman (root `my_team.py`) | 941f89c |
| `vsi7_m.py` | https://github.com/vsi7-m/despookjes | ff31ee6 |

Left out:
- In-The-Fridge (Wakie26) crashes every game with this framework version.
- pacman-agent-BonaTarda (Cambie26) crashed in 6 of 16 games.
- tortosa_pacman and pacboywhite are not publicly reachable.

Known crashes on their side (the eval marks these games CRASH): `foxtrotdelta_hub` on
bloxCapture, `vsi7_m` occasionally on random mazes.
