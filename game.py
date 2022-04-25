import random
import tensorflow as tf
import pygame
from typing import Tuple, List, Sequence
from pygame.surface import Surface

pygame.init()


# =======================================classes==============================================

class NN:
    """
    Neural network used to control movement of paddles. Options: up, down, no movement
    """

    def __init__(self, path: str):
        """
        Constructor
        :param path: path to the model
        """
        self.model = tf.keras.models.load_model(path)

    def predict(self, delta_x, delta_y, x_vel, y_vel) -> int:
        """
        Method used to predict outputs.
        0 - no movement
        1 - up
        2 - down
        :param delta_x: x distance between the paddle and the ball
        :param delta_y: y distance between the paddle and the ball
        :param x_vel: x parameter of the ball velocity
        :param y_vel: y parameter of the ball velocity
        """
        data = tf.convert_to_tensor([[delta_x, delta_y, x_vel, y_vel]])
        pred = self.model.predict(data)
        return self.transform(pred[0])

    def transform(self, probabilities) -> int:
        """
        Inner method used to convert probabilities into numbers
        :param probabilities: list of output probabilities generated by neural network
        """
        if probabilities[0] > probabilities[1] and probabilities[0] > probabilities[2]:
            return 0
        elif probabilities[1] > probabilities[2]:
            return 1
        else:
            return 2


class NN_no_stops:
    """
    Neural network used to control movement of paddles
    """

    def __init__(self, path):
        self.model = tf.keras.models.load_model(path)

    def predict(self, delta_x, delta_y, x_vel, y_vel) -> int:
        """
        Method used to predict outputs.
        1 - up
        2 - down
        :param delta_x: x distance between the paddle and the ball
        :param delta_y: y distance between the paddle and the ball
        :param x_vel: x parameter of the ball velocity
        :param y_vel: y parameter of the ball velocity
        """
        data = tf.convert_to_tensor([[delta_x, delta_y, x_vel, y_vel]])
        pred = self.model.predict(data)
        return self.transform(pred[0])

    def transform(self, probabilities) -> int:
        """
        Inner method used to convert probabilities into numbers
        :param probabilities: list of output probabilities generated by neural network
        """
        if probabilities[0] > probabilities[1]:
            return 1
        else:
            return 2


class Paddle:
    """
    A paddle class
    """
    VELOCITY = 4

    def __init__(self, x: int, y: int, width: int, height: int, color: Tuple[int, int, int] = (255, 255, 255)):
        """
        Construction method
        :param x: Initial x position
        :param y: Initial y position
        :param width: width of the paddle
        :param height: height of the paddle
        :param color: color of the paddle
        """
        self.x = x
        self.y = y
        self.init_x = x
        self.init_y = y
        self.width = width
        self.height = height
        self.color = color

    def draw(self, win: Surface) -> None:
        """
        Function used to draw the paddle
        :param win: window where to draw
        """
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height))

    def move(self, up: bool = True) -> None:
        """
        Function used to move the paddle upwards or downwards
        :param up: if true, then the paddle will move upwards
        """
        if up:
            self.y -= self.VELOCITY
        else:
            self.y += self.VELOCITY

    def reset(self) -> None:
        """
        Function used to set the paddle to initial position
        """
        self.x = self.init_x
        self.y = self.init_y


class Ball:
    """
    Ball class
    """

    def __init__(self, x: int, y: int, radius: int, color: Tuple[int, int, int] = (255, 255, 255), max_vel: int = 6):
        """
        Constructor
        :param x: Initial x coordinate
        :param y: Initial y coordinate
        :param radius: Radius of the ball
        :param color: Color of the ball
        :param max_vel: Maximum average velocity that ball can reach
        """
        self.MAX_VEL = max_vel
        self.x = x
        self.y = y
        self.init_x = x
        self.init_y = y
        self.radius = radius
        if random.choice([True, False]):
            self.x_vel = random.randint(max_vel // 1.5, max_vel)
        else:
            self.x_vel = -1 * random.randint(max_vel // 1.5, max_vel)
        if random.choice([True, False]):
            self.y_vel = random.randint(0, max_vel)
        else:
            self.y_vel = -1 * random.randint(0, max_vel)
        self.color = color

    def draw(self, win: Surface) -> None:
        """
        Function used to draw the ball
        :param win: Window where to draw the ball
        """
        pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)

    def move(self) -> None:
        """
        Function used to move the ball according to the current velocity
        """
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self) -> None:
        """
        Function used to set the ball to initial position
        """
        self.x = self.init_x
        self.y = self.init_y
        if random.choice([True, False]):
            self.x_vel = random.randint(self.MAX_VEL // 1.5, self.MAX_VEL)
        else:
            self.x_vel = -1 * random.randint(self.MAX_VEL // 1.5, self.MAX_VEL)
        if random.choice([True, False]):
            self.y_vel = random.randint(0, self.MAX_VEL)
        else:
            self.y_vel = -1 * random.randint(0, self.MAX_VEL)

class Game:

    def __init__(self,
                 paddle_width: int = 20,
                 paddle_height: int = 100,
                 ball_size: int = 7,
                 win_width=800,
                 win_height=600,
                 fps=60):
        """
        Constructor method
        :param paddle_width: width of the paddle; default = 20
        :param paddle_height: height of the paddle; default = 100
        :param ball_size: radius of the ball; default = 7
        :param win_width: width od the window; default = 800
        :param win_height: height of the window; default = 600
        :param fps: frames per second; default = 60
        """
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.PADDLE_WIDTH: int = paddle_width
        self.PADDLE_HEIGHT: int = paddle_height
        self.BALL_SIZE: int = ball_size
        self.WIDTH = win_width
        self.HEIGHT = win_height
        self.FPS = fps
        self.WIN = pygame.display.set_mode((win_width, win_height))
        pygame.display.set_caption("Pong")
        self.SCORE_FONT = pygame.font.SysFont("comicsans", 50)

    # =======================================controls==============================================

    def paddle_movement_with_if_control(self, paddle: Paddle, ball: Ball, path: str) -> None:
        with open(path, "a+") as f:
            if ball.y < paddle.y + (paddle.height // 2) and paddle.y - paddle.VELOCITY >= 0:
                paddle.move(up=True)
                f.write(f"{paddle.x},{paddle.y},{ball.x},{ball.y},{ball.x_vel},{ball.y_vel},{1}\n")
            elif paddle.y + paddle.VELOCITY + paddle.height <= self.HEIGHT:
                paddle.move(up=False)
                f.write(f"{paddle.x},{paddle.y},{ball.x},{ball.y},{ball.x_vel},{ball.y_vel},{2}\n")
            else:
                f.write(f"{paddle.x},{paddle.y},{ball.x},{ball.y},{ball.x_vel},{ball.y_vel},{0}\n")

    def paddle_movement(self, keys: Sequence[bool], left_paddle: Paddle, right_paddle: Paddle) -> None:
        """
        Function used to control movement of the paddles
        :param keys: All the inputs of a keyboard
        :param left_paddle: Left paddle
        :param right_paddle: Right paddle
        """
        if keys[pygame.K_w] and left_paddle.y - left_paddle.VELOCITY >= 0:
            left_paddle.move(up=True)
        if keys[pygame.K_s] and left_paddle.y + left_paddle.VELOCITY + left_paddle.height <= self.HEIGHT:
            left_paddle.move(up=False)
        if keys[pygame.K_UP] and right_paddle.y - right_paddle.VELOCITY >= 0:
            right_paddle.move(up=True)
        if keys[pygame.K_DOWN] and right_paddle.y + right_paddle.VELOCITY + right_paddle.height <= self.HEIGHT:
            right_paddle.move(up=False)

        # =======================================collisions==============================================

    def collisions_with_random(self, ball: Ball, left_paddle: Paddle, right_paddle: Paddle) -> None:
        """
        Function used to handle the collisions of the ball and paddles.
        This version of the function allows the ball to randomly change its velocity after collisions with the paddles.
        :param ball: Ball
        :param left_paddle: Left paddle
        :param right_paddle: Right paddle
        """
        if ball.y + ball.radius >= self.HEIGHT:
            ball.y_vel *= -1
        elif ball.y - ball.radius <= 0:
            ball.y_vel *= -1

        if ball.x_vel < 0:
            if left_paddle.y <= ball.y <= left_paddle.y + left_paddle.height and \
                    ball.x - ball.radius <= left_paddle.x + left_paddle.width:
                ball.x_vel *= -1
                if random.choice([True, False]):
                    rand_y = random.random() * (ball.MAX_VEL / 10)
                else:
                    rand_y = -1 * random.random() * (ball.MAX_VEL / 10)
                ball.y_vel += rand_y
                ball.x_vel *= 1.02
        else:
            if right_paddle.y <= ball.y <= right_paddle.y + right_paddle.height and \
                    ball.x + ball.radius >= right_paddle.x:
                ball.x_vel *= -1
                if random.choice([True, False]):
                    rand_y = random.random() * (ball.MAX_VEL / 10)
                else:
                    rand_y = -1 * random.random() * (ball.MAX_VEL / 10)
                ball.y_vel += rand_y
                ball.x_vel *= 1.02

    def collisions_no_random(self, ball: Ball, left_paddle: Paddle, right_paddle: Paddle) -> None:
        """
        Function used to handle the collisions of the ball and paddles.
        This version of the function changes the y-component of the velocity of the ball according to the position of the
        paddle where the collision happened.
        :param ball: Ball
        :param left_paddle: Left paddle
        :param right_paddle: Right paddle
        """
        if ball.y + ball.radius >= self.HEIGHT:
            ball.y_vel *= -1
        elif ball.y - ball.radius <= 0:
            ball.y_vel *= -1

        if ball.x_vel < 0:
            if left_paddle.y <= ball.y <= left_paddle.y + left_paddle.height and \
                    ball.x - ball.radius <= left_paddle.x + left_paddle.width:
                ball.x_vel *= -1

                middle_y = left_paddle.y + left_paddle.height / 2
                difference_in_y = middle_y - ball.y
                reduction_factor = (left_paddle.height / 2) / ball.MAX_VEL
                y_vel = difference_in_y / reduction_factor
                ball.y_vel = -1 * y_vel
                ball.x_vel *= 1.02
        else:
            if right_paddle.y <= ball.y <= right_paddle.y + right_paddle.height and \
                    ball.x + ball.radius >= right_paddle.x:
                ball.x_vel *= -1
                middle_y = right_paddle.y + right_paddle.height / 2
                difference_in_y = middle_y - ball.y
                reduction_factor = (right_paddle.height / 2) / ball.MAX_VEL
                y_vel = difference_in_y / reduction_factor
                ball.y_vel = -1 * y_vel
                ball.x_vel *= 1.02

    # =======================================collecting data==============================================

    def collect_data(self, keys: Sequence[bool], ball: Ball, paddle: Paddle, path: str) -> None:
        """
        Function used to collect the data of how the player plays the game.
        The resulting file is in the .csv format.
        Columns: paddle_x, paddle_y, ball_x, ball_y, ball_x_vel, ball_y_vel, label
        Labels: 0 - no movement, 1 - up, 2 - down
        :param keys: All the inputs of a keyboard
        :param ball: Ball for which the data will be collected
        :param paddle: Paddle for which the data will be collected
        :param path: path for the csv file
        """
        with open(path, "a+") as f:
            if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
                f.write(f"{paddle.x},{paddle.y},{ball.x},{ball.y},{ball.x_vel},{ball.y_vel},{1}\n")
            elif keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
                f.write(f"{paddle.x},{paddle.y},{ball.x},{ball.y},{ball.x_vel},{ball.y_vel},{2}\n")
            else:
                f.write(f"{paddle.x},{paddle.y},{ball.x},{ball.y},{ball.x_vel},{ball.y_vel},{0}\n")

    def collect_data_mod(self, keys: Sequence[bool], ball: Ball, paddle: Paddle, path: str):
        """
        Function used to collect the data of how the player plays the game.
        This version of the data has fewer inputs to simplify the training process
        The resulting file is in the .csv format.
        Columns: delta_x, delta_y, ball_x_vel, ball_y_vel, label
        Labels: 1 - up, 2 - down
        :param keys: All the inputs of a keyboard
        :param ball: Ball for which the data will be collected
        :param paddle: Paddle for which the data will be collected
        :param path: path for the csv file
        """
        with open(path, "a+") as f:
            if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
                f.write(
                    f"{paddle.x - ball.x},{paddle.y + (paddle.height // 2) - ball.y},{ball.x_vel},{ball.y_vel},{1}\n")
            elif keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
                f.write(
                    f"{paddle.x - ball.x},{paddle.y + (paddle.height // 2) - ball.y},{ball.x_vel},{ball.y_vel},{2}\n")
            else:
                f.write(
                    f"{paddle.x - ball.x},{paddle.y + (paddle.height // 2) - ball.y},{ball.x_vel},{ball.y_vel},{0}\n")

    # =======================================drawing==============================================

    def draw(self, win: Surface, paddles: List[Paddle], ball: Ball, left_score: int, right_score: int) -> None:
        """
        Function used to draw all the objects in the window
        :param win: Window where to draw
        :param paddles: List of paddles
        :param ball: Ball
        :param left_score: Score of the left player
        :param right_score: Score of the right player
        """
        win.fill(self.BLACK)

        left_score_text = self.SCORE_FONT.render(f"{left_score}", 1, self.WHITE)
        right_score_text = self.SCORE_FONT.render(f"{right_score}", 1, self.WHITE)
        win.blit(left_score_text, (self.WIDTH // 4 - left_score_text.get_width() // 2, 20))
        win.blit(right_score_text, (self.WIDTH // 4 * 3 - right_score_text.get_width() // 2, 20))

        for paddle in paddles:
            paddle.draw(win)

        ball.draw(win)

        pygame.display.update()

    # =======================================main loops==============================================

    def main_collect_data(self) -> None:
        """
        Main loop of the game without AI controlling the paddles
        """
        run = True
        clock = pygame.time.Clock()

        # temporal variables to set size of the left paddle for training
        temp_paddle_height = 600
        temp_paddle_width = 20
        right_paddle = Paddle(self.WIDTH - 10 - self.PADDLE_WITH, self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2,
                              self.PADDLE_WITH, self.PADDLE_HEIGHT)
        left_paddle = Paddle(10, self.HEIGHT // 2 - temp_paddle_height // 2, temp_paddle_width, temp_paddle_height)
        # left_paddle = Paddle(10, self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        # right_paddle = Paddle(self.WIDTH - 10 - temp_paddle_width, self.HEIGHT // 2 - temp_paddle_height // 2,
        #                       self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        ball = Ball(self.WIDTH // 2, self.HEIGHT // 2, self.BALL_SIZE)

        left_score = 0
        right_score = 0

        while run:
            self.draw(self.WIN, [left_paddle, right_paddle], ball, left_score, right_score)
            clock.tick(self.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            keys = pygame.key.get_pressed()
            self.paddle_movement(keys, left_paddle, right_paddle)

            ball.move()

            # data collection
            path = "data_example.txt"
            # collect_data_mod(keys, ball, right_paddle, path)
            self.paddle_movement_with_if_control(right_paddle, ball, path)

            # random
            self.collisions_with_random(ball, left_paddle, right_paddle)
            # collisions_no_random(ball, left_paddle, right_paddle)

            if ball.x < left_paddle.width // 10:
                right_score += 1
                ball.reset()
                left_paddle.reset()
                right_paddle.reset()
            if ball.x > self.WIDTH - right_paddle.width // 10:
                left_score += 1
                ball.reset()
                left_paddle.reset()
                right_paddle.reset()

            if pygame.key.get_pressed()[pygame.K_r]:
                ball.reset()
                left_paddle.reset()
                right_paddle.reset()
                left_score = 0
                right_score = 0

        pygame.quit()

    def main_ai(self) -> None:
        """
        Main loop of the game with AI controlling one paddle
        """
        run = True
        clock = pygame.time.Clock()

        left_paddle = Paddle(10, self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        right_paddle = Paddle(self.WIDTH - 10 - self.PADDLE_WIDTH, self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2,
                              self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        ball = Ball(self.WIDTH // 2, self.HEIGHT // 2, self.BALL_SIZE)

        left_score = 0
        right_score = 0

        # ai = NN('model_if')ss
        ai = NN_no_stops('model_if_no_zeroes')

        while run:
            self.draw(self.WIN, [left_paddle, right_paddle], ball, left_score, right_score)
            clock.tick(self.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            keys = pygame.key.get_pressed()
            self.paddle_movement(keys, left_paddle, right_paddle)

            delta_x = right_paddle.x - ball.x
            delta_y = right_paddle.y - ball.y - right_paddle.height // 2
            output = ai.predict(delta_x, delta_y, ball.x_vel, ball.y_vel) + 1
            # output = ai.predict(delta_x, delta_y, ball.x_vel, ball.y_vel)
            if output == 1 and right_paddle.y - right_paddle.VELOCITY >= 0:
                right_paddle.move(up=True)
            elif output == 2 and right_paddle.y + right_paddle.VELOCITY + right_paddle.height <= self.HEIGHT:
                right_paddle.move(up=False)
            ball.move()

            # random
            self.collisions_with_random(ball, left_paddle, right_paddle)
            # collisions_no_random(ball, left_paddle, right_paddle)

            if ball.x < left_paddle.width // 5:
                right_score += 1
                ball.reset()
                left_paddle.reset()
                right_paddle.reset()
            if ball.x > self.WIDTH - right_paddle.width // 5:
                left_score += 1
                ball.reset()
                left_paddle.reset()
                right_paddle.reset()

            if pygame.key.get_pressed()[pygame.K_r]:
                ball.reset()
                left_paddle.reset()
                right_paddle.reset()
                left_score = 0
                right_score = 0

        pygame.quit()

    def main(self):
        """
        Main loop
        """
        run = True
        clock = pygame.time.Clock()

        left_paddle = Paddle(10, self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        right_paddle = Paddle(self.WIDTH - 10 - self.PADDLE_WIDTH, self.HEIGHT // 2 - self.PADDLE_HEIGHT // 2,
                              self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        ball = Ball(self.WIDTH // 2, self.HEIGHT // 2, self.BALL_SIZE)

        left_score = 0
        right_score = 0

        while run:
            self.draw(self.WIN, [left_paddle, right_paddle], ball, left_score, right_score)
            clock.tick(self.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            keys = pygame.key.get_pressed()
            self.paddle_movement(keys, left_paddle, right_paddle)

            ball.move()

            # random
            self.collisions_with_random(ball, left_paddle, right_paddle)
            # collisions_no_random(ball, left_paddle, right_paddle)

            if ball.x < left_paddle.width // 5:
                right_score += 1
                ball.reset()
                left_paddle.reset()
                right_paddle.reset()
            if ball.x > self.WIDTH - right_paddle.width // 5:
                left_score += 1
                ball.reset()
                left_paddle.reset()
                right_paddle.reset()

            if pygame.key.get_pressed()[pygame.K_r]:
                ball.reset()
                left_paddle.reset()
                right_paddle.reset()
                left_score = 0
                right_score = 0

        pygame.quit()


if __name__ == "__main__":
    game = Game()
    game.main()
    # game.main_ai()