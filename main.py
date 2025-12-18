import pygame

INIT_WINDOW_SIZE = (1280, 720)
FPS = 60


def set_windowed():
    """Sets the pygame display window to a 1280x720 window."""
    return pygame.display.set_mode((1280, 720), pygame.RESIZABLE)

def set_fullscreen():
    """Sets the pygame display window to fullscreen mode."""
    info = pygame.display.Info()
    size = (info.current_w, info.current_h)
    return pygame.display.set_mode((size), pygame.RESIZABLE)


def main():
    # Initializes Pygame
    pygame.init()

    # info = pygame.display.Info()
    # DISPLAY_MAX_SIZE = (info.current_w, info.current_h)

    # Sets up the game display window
    fullscreen = False
    screen = set_windowed()

    pygame.display.set_caption("Kingdom Survival")

    clock = pygame.time.Clock()
    running = True

    # Game loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.fill((0, 0, 0))

        pygame.display.flip()
        clock.tick(FPS)

    # Quits Pygame
    pygame.quit()
    return

if __name__ == "__main__":
    main()