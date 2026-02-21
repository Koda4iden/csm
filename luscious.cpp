#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <conio.h>
#include <windows.h>
#else
#include <chrono>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#endif

struct Particle {
    float x;
    float y;
    float vx;
    float vy;
    bool is_foam;
    float aeration;
};

class LusciousConstraint {
private:
    int width = 70;
    int height = 35;
    float cream_viscosity = 0.085f;
    float frother_agitation = 7.5f;
    float frother_frequency = 0.25f;
    float container_responsiveness = 0.92f;

    std::vector<Particle> particles;
    int foam_count = 0;
    int total_steps = 0;

    void clear_screen() {
#if defined(_WIN32)
        std::system("cls");
#else
        std::cout << "\x1B[2J\x1B[H";
#endif
    }

    static bool key_pressed() {
#if defined(_WIN32)
        return _kbhit();
#else
        termios oldt;
        tcgetattr(STDIN_FILENO, &oldt);
        termios newt = oldt;
        newt.c_lflag &= static_cast<unsigned int>(~(ICANON | ECHO));
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);

        int oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

        int ch = getchar();

        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        fcntl(STDIN_FILENO, F_SETFL, oldf);

        return ch != EOF;
#endif
    }

    void inject_frother_energy() {
        if (static_cast<float>(std::rand()) / RAND_MAX < frother_frequency) {
            int idx = std::rand() % particles.size();
            particles[idx].vx += (static_cast<float>(std::rand() % 100) / 50.0f - 1.0f) * frother_agitation;
            particles[idx].vy += (static_cast<float>(std::rand() % 100) / 50.0f - 1.0f) * frother_agitation;
        }
    }

public:
    LusciousConstraint() {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        for (int i = 0; i < 1200; ++i) {
            float x = 5.0f + static_cast<float>(std::rand() % (width - 10));
            float y = 3.0f + static_cast<float>(std::rand() % (height - 6));
            particles.push_back({x, y, 0.0f, 0.0f, false, 0.0f});
        }
    }

    void set_parameters(float visc, float agit, float freq) {
        cream_viscosity = visc;
        frother_agitation = agit;
        frother_frequency = freq;
    }

    void update() {
        total_steps++;
        inject_frother_energy();

        foam_count = 0;
        for (auto& p : particles) {
            p.vx *= (1.0f - cream_viscosity);
            p.vy *= (1.0f - cream_viscosity);

            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 2.0f || p.x > width - 3.0f) {
                p.vx *= -container_responsiveness;
            }
            if (p.y < 2.0f || p.y > height - 3.0f) {
                p.vy *= -container_responsiveness;
            }
            if (p.x < 0.0f) {
                p.x = 0.0f;
            }
            if (p.x > static_cast<float>(width)) {
                p.x = static_cast<float>(width);
            }
            if (p.y < 0.0f) {
                p.y = 0.0f;
            }
            if (p.y > static_cast<float>(height)) {
                p.y = static_cast<float>(height);
            }

            float speed = std::sqrt(p.vx * p.vx + p.vy * p.vy);
            if (speed < 1.8f && !p.is_foam) {
                p.is_foam = true;
                p.aeration = 1.0f - speed / 1.8f;
            }
            if (p.is_foam) {
                foam_count++;
            }
        }
    }

    void render() {
        clear_screen();
        std::vector<std::string> grid(height, std::string(width, ' '));

        for (int x = 0; x < width; ++x) {
            grid[0][x] = '=';
            grid[height - 1][x] = '=';
        }
        for (int y = 0; y < height; ++y) {
            grid[y][0] = '|';
            grid[y][width - 1] = '|';
        }

        for (const auto& p : particles) {
            int ix = static_cast<int>(p.x);
            int iy = static_cast<int>(p.y);
            if (ix < 0 || ix >= width || iy < 0 || iy >= height) {
                continue;
            }

            if (p.is_foam) {
                grid[iy][ix] = (p.aeration > 0.7f) ? '@' : 'o';
            } else if (std::abs(p.vx) + std::abs(p.vy) > 2.5f) {
                grid[iy][ix] = '*';
            } else {
                grid[iy][ix] = '.';
            }
        }

        std::cout << "=== LUSCIOUS CONSTRAINT : Frother Dipped In Heavy Cream ===\n";
        for (const auto& row : grid) {
            std::cout << row << '\n';
        }
        std::cout << "============================================================\n";
        std::cout << "Step: " << total_steps
                  << " | Foam particles: " << foam_count
                  << " (" << (foam_count * 100 / static_cast<int>(particles.size())) << "%)"
                  << " | Cream viscosity: " << cream_viscosity
                  << " | Frother agitation: " << frother_agitation << '\n';

        std::string state = (foam_count > 650) ? "LUSCIOUS & SHIMMERING"
                           : (foam_count > 400) ? "TRANSFORMING INTO FOAM"
                                                : "AGITATING - STILL BECOMING";
        std::cout << "Relationship state: " << state << '\n';
    }

    bool should_continue() const {
        return !key_pressed();
    }
};

int main() {
    LusciousConstraint sim;
    sim.set_parameters(0.085f, 7.5f, 0.25f);

    std::cout << "Luscious Constraint running...\n";
    std::cout << "Watch chaos become held, held become luminous.\n";
    std::cout << "Press any key to stop.\n\n";

    while (sim.should_continue()) {
        sim.update();
        sim.render();
#if defined(_WIN32)
        Sleep(60);
#else
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
#endif
    }

    std::cout << "\nThe container still holds. The foam remains.\n";
    return 0;
}
