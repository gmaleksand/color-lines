import sys

import numpy as np
import pygame
from pygame.locals import QUIT, KEYDOWN

from env import EnvNext3    # the game itself
import policy

CELL = 90
PANEL_W = 150
W, H = 9 * CELL + 2 * CELL, 9 * CELL + 2 * CELL   # 990 x 990 board area


def colorize(num):
    return {
        0: (128, 128, 128),   # empty (zero-padded next3 slot)
        1: (255, 0, 0),       # RED
        2: (0, 255, 0),       # GREEN
        3: (0, 0, 255),       # BLUE
        4: (255, 255, 0),     # YELLOW
        5: (255, 0, 255),     # PURPLE
        6: (0, 255, 255),     # CYAN
        7: (255, 255, 255),   # WHITE
        8: (0, 0, 0),         # border
    }[num]


# --------------------------------------------------------------------------
# game logic (display-independent; used by the GUI)
# --------------------------------------------------------------------------

class Game:
    """One interactive game: env state.  All move application
    goes through here."""

    def __init__(self, seed):
        self.seed = int(seed)
        self.env = EnvNext3()
        self.env.reset(seed=self.seed)
        self.policy = policy.Policy()
        self.step = 0
        self._overlay = None

    # -- strategy side -----------------------------------------------------

    def move(self):
        a = self.policy.get_action(self.env.board.copy(), self.env.next3.copy())
        return None if a is None else (int(a[0]), int(a[1]))

    # -- human side --------------------------------------------------------

    def valid_dests(self, src_idx):
        return self.env.get_valid_dests_for_source(src_idx)

    def apply_move(self, src, dst):
        """Apply the human's move; return whether the game is over."""
        _, done, _ = self.env.step(src * 81 + dst)
        self.step += 1
        return done


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

def draw_board(DISPLAYSURF, board):
    for i in range(11):
        for j in range(11):
            if board[i][j] == 8:
                pygame.draw.rect(DISPLAYSURF, colorize(8), (i * CELL, j * CELL, CELL, CELL))
            elif board[i][j] == 0:
                pygame.draw.rect(DISPLAYSURF, colorize(0), (i * CELL, j * CELL, CELL, CELL))
            else:
                pygame.draw.circle(DISPLAYSURF, colorize(board[i][j]),
                                   (i * CELL + 45, j * CELL + 45), 40)
            pygame.draw.rect(DISPLAYSURF, (0, 0, 0), (i * CELL, j * CELL, CELL, CELL), width=1)


def draw_panel(DISPLAYSURF, font, font_small, game, score, notice=False):
    x0 = W
    pygame.draw.rect(DISPLAYSURF, (40, 40, 40), (x0, 0, PANEL_W, H))
    surf = font.render('NEXT 3', True, (255, 255, 255))
    DISPLAYSURF.blit(surf, (x0 + 18, 16))
    for k in range(3):
        c = int(game.env.next3[k])
        cy = 90 + k * 78
        pygame.draw.circle(DISPLAYSURF, colorize(c), (x0 + PANEL_W // 2, cy), 32)
        pygame.draw.circle(DISPLAYSURF, (255, 255, 255), (x0 + PANEL_W // 2, cy), 32, width=2)

    def label(text, y, fnt, color=(255, 255, 255)):
        s = fnt.render(text, True, color)
        DISPLAYSURF.blit(s, (x0 + 14, y))

    label('SCORE', 360, font_small, (180, 180, 180))
    label(str(score), 382, font)
    label('MOVES', 450, font_small, (180, 180, 180))
    label(str(game.step), 472, font)
    label('SEED', 540, font_small, (180, 180, 180))
    s = font_small.render(str(game.seed), True, (180, 180, 180))
    DISPLAYSURF.blit(s, (x0 + 12, 562))
    label('R: replay seed', 780, font_small, (120, 120, 120))
    label('click a ball', 830, font_small, (120, 120, 120))
    label('then a destination', 846, font_small, (120, 120, 120))
    if notice:
        label('policy compile', 620, font_small, (255, 200, 100))
        label('please wait', 646, font_small, (255, 200, 100))


def draw_arrow(DISPLAYSURF, src, dst):
    start = np.array([(src // 9) * CELL + 135, (src % 9) * CELL + 135])
    end = np.array([(dst // 9) * CELL + 135, (dst % 9) * CELL + 135])
    vecn = (end - start) / np.linalg.norm(end - start)
    vecp = np.array([-vecn[1], vecn[0]])
    pygame.draw.polygon(DISPLAYSURF, (255, 255, 255), (
        start + 5 * vecp, start - 5 * vecp, end - 26 * vecn - 5 * vecp,
        end - 39 * vecn - 26 * vecp, end, end - 39 * vecn + 26 * vecp,
        end - 26 * vecn + 5 * vecp))


def build_gameover_overlay(game):
    """Render the game-over overlay once; returns the surface."""
    lines = [
        ('GAME OVER', 46, (255, 255, 255)),
        (f"your score: {game.env.score}", 30, (255, 255, 255)),
        ('click: new game (random seed)      R: replay this seed', 22, (200, 200, 200)),
    ]
    ov = pygame.Surface((W, H), pygame.SRCALPHA)
    ov.fill((0, 0, 0, 170))
    y = H // 2 - 130
    for text, size, col in lines:
        s = pygame.font.Font('freesansbold.ttf', size).render(text, True, col)
        ov.blit(s, (W // 2 - s.get_width() // 2, y))
        y += size + 18
    return ov


def run_gui(seed):
    pygame.init()
    font = pygame.font.Font('freesansbold.ttf', 30)
    font_small = pygame.font.Font('freesansbold.ttf', 18)
    DISPLAYSURF = pygame.display.set_mode((W + PANEL_W, H))
    pygame.display.set_caption('Color Lines - next-3 colors exposed')

    game = Game(seed)
    done = False
    selecting = False
    sel_src = None
    sel_bg = None      # saved cell background under the selection ring
    sel_rect = None
    turn_data = None   # whether this turn is already drawn
    compiled = False   # policy has already been compiled

    def clear_selection():
        nonlocal selecting, sel_src, sel_bg, sel_rect
        if sel_bg is not None:
            DISPLAYSURF.blit(sel_bg, sel_rect)
        selecting, sel_src, sel_bg, sel_rect = False, None, None, None

    def new_game(new_seed):
        nonlocal game, done, turn_data
        clear_selection()
        game = Game(new_seed)
        done = False
        turn_data = None

    while True:
        # ---- new turn: redraw board + panel, show it, then compute & draw arrow ----
        if turn_data is None:
            DISPLAYSURF.fill((128, 128, 128))
            bdisp = np.zeros((11, 11), dtype=int)
            for u in range(11):
                bdisp[0][u] = bdisp[10][u] = bdisp[u][0] = bdisp[u][10] = 8
            bdisp[1:-1, 1:-1] = game.env.board
            draw_board(DISPLAYSURF, bdisp)
            draw_panel(DISPLAYSURF, font, font_small, game, game.env.score,
                       notice=not compiled)

            # Show the current state before the policy runs / compiles.
            pygame.display.update()

            mv = game.move()
            if not compiled:
                compiled = True
                # remove the "please wait" notice once compilation finishes
                draw_panel(DISPLAYSURF, font, font_small, game, game.env.score)
                pygame.display.update()

            if mv is None:
                done = True
                turn_data = ()
            else:
                turn_data = mv
                draw_arrow(DISPLAYSURF, *mv)
                pygame.display.update()

            sel_bg, sel_rect = None, None

        if done:
            if game._overlay is None:
                game._overlay = build_gameover_overlay(game)
            DISPLAYSURF.blit(game._overlay, (0, 0))
            pygame.display.update()

        # ---- events -------------------------------------------------------
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == KEYDOWN and event.key == pygame.K_r and done:
                new_game(game.seed)
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if done:
                    new_game(int(np.random.randint(1, 2 ** 63 - 1)))
                    continue
                if x >= W:          # panel clicks are ignored
                    continue
                i, j = int(x / CELL) - 1, int(y / CELL) - 1
                if not (0 <= i < 9 and 0 <= j < 9):
                    continue
                idx = i * 9 + j
                b = game.env.board
                if not selecting:
                    if b[i, j] != 0:
                        sel_bg = DISPLAYSURF.subsurface(
                            pygame.Rect((i + 1) * CELL, (j + 1) * CELL, CELL, CELL)).copy()
                        sel_rect = pygame.Rect((i + 1) * CELL, (j + 1) * CELL, CELL, CELL)
                        sel_src = idx
                        selecting = True
                        pygame.draw.circle(DISPLAYSURF, (0, 0, 0),
                                           ((i + 1) * CELL + 45, (j + 1) * CELL + 45), 42, width=3)
                        pygame.display.update()
                else:
                    if idx == sel_src:
                        clear_selection()
                        pygame.display.update()
                    elif b[i, j] != 0:
                        # switch selection to the newly clicked ball
                        clear_selection()
                        sel_bg = DISPLAYSURF.subsurface(
                            pygame.Rect((i + 1) * CELL, (j + 1) * CELL, CELL, CELL)).copy()
                        sel_rect = pygame.Rect((i + 1) * CELL, (j + 1) * CELL, CELL, CELL)
                        sel_src = idx
                        selecting = True
                        pygame.draw.circle(DISPLAYSURF, (0, 0, 0),
                                           ((i + 1) * CELL + 45, (j + 1) * CELL + 45), 42, width=3)
                        pygame.display.update()
                    elif game.valid_dests(sel_src)[idx]:
                        s = sel_src
                        clear_selection()
                        done = game.apply_move(s, idx)
                        turn_data = None   # force full redraw next loop
                    else:
                        clear_selection()
                        pygame.display.update()

        pygame.time.Clock().tick(60)


if __name__ == '__main__':
    seed = int(np.random.randint(1, 2 ** 63 - 1))
    run_gui(seed)
