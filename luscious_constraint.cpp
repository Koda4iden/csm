#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct FrotherConfig {
    double intensity = 3.0;
    double frequency = 0.45;
    int durationSteps = 700;
    double radius = 4.0;
    double noise = 0.25;
};

struct CreamConfig {
    double viscosity = 0.36;
    double density = 1.3;
    double responsiveness = 0.75;
};

struct ContainerConfig {
    int width = 72;
    int height = 34;
    std::string boundary = "reflective"; // reflective or absorbent
};

struct SimConfig {
    FrotherConfig frother;
    CreamConfig cream;
    ContainerConfig container;
    int totalSteps = 1400;
    int foamSmoothingWindow = 18;
    int snapshotEvery = 140;
    std::uint32_t seed = 7;
    std::string csvPath = "foam_metrics.csv";
};

struct CellState {
    double energy = 0.0;
    double foam = 0.0;
};

class LusciousConstraintSim {
  public:
    explicit LusciousConstraintSim(const SimConfig &cfg)
        : cfg_(cfg), grid_(cfg.container.width * cfg.container.height),
          scratch_(cfg.container.width * cfg.container.height), rng_(cfg.seed),
          noiseDist_(-1.0, 1.0) {
        if (cfg_.container.width < 5 || cfg_.container.height < 5) {
            throw std::invalid_argument("Container must be at least 5x5.");
        }
        if (cfg_.foamSmoothingWindow <= 0 || cfg_.snapshotEvery <= 0 ||
            cfg_.totalSteps <= 0) {
            throw std::invalid_argument("Step/window values must be positive.");
        }
    }

    void run() {
        std::ofstream csv(cfg_.csvPath);
        csv << "step,mean_energy,max_energy,foam_mass,foam_clusters\n";

        for (int step = 0; step < cfg_.totalSteps; ++step) {
            injectFrother(step);
            diffuseAndDamp(step);
            updateFoam();

            if (step % cfg_.snapshotEvery == 0 || step == cfg_.totalSteps - 1) {
                Metrics m = measure();
                csv << step << ',' << m.meanEnergy << ',' << m.maxEnergy << ','
                    << m.foamMass << ',' << m.foamClusters << "\n";
                std::cout << renderSnapshot(step, m) << "\n";
            }
        }

        std::cout << "Wrote metrics to: " << cfg_.csvPath << "\n";
    }

  private:
    struct Metrics {
        double meanEnergy;
        double maxEnergy;
        double foamMass;
        int foamClusters;
    };

    SimConfig cfg_;
    std::vector<CellState> grid_;
    std::vector<CellState> scratch_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> noiseDist_;

    int idx(int x, int y) const { return y * cfg_.container.width + x; }

    bool inBounds(int x, int y) const {
        return x >= 0 && y >= 0 && x < cfg_.container.width &&
               y < cfg_.container.height;
    }

    void injectFrother(int step) {
        if (step > cfg_.frother.durationSteps) {
            return;
        }

        const double cx = cfg_.container.width / 2.0 +
                          std::sin(step * cfg_.frother.frequency * 0.35) *
                              (cfg_.container.width * 0.17);
        const double cy = cfg_.container.height / 2.0 +
                          std::cos(step * cfg_.frother.frequency * 0.29) *
                              (cfg_.container.height * 0.16);

        const int minX = static_cast<int>(std::floor(cx - cfg_.frother.radius));
        const int maxX = static_cast<int>(std::ceil(cx + cfg_.frother.radius));
        const int minY = static_cast<int>(std::floor(cy - cfg_.frother.radius));
        const int maxY = static_cast<int>(std::ceil(cy + cfg_.frother.radius));

        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                if (!inBounds(x, y)) {
                    continue;
                }
                const double dx = x - cx;
                const double dy = y - cy;
                const double dist = std::sqrt(dx * dx + dy * dy);
                if (dist > cfg_.frother.radius) {
                    continue;
                }

                const double radial = 1.0 - (dist / cfg_.frother.radius);
                const double wave = 0.5 +
                                    0.5 * std::sin(step * cfg_.frother.frequency +
                                                   dist * 2.2);
                const double noise = cfg_.frother.noise * noiseDist_(rng_);
                const double impulse = cfg_.frother.intensity * radial *
                                       (wave + 0.2 + noise) /
                                       std::max(0.1, cfg_.cream.density);

                grid_[idx(x, y)].energy += std::max(0.0, impulse);
            }
        }
    }

    void diffuseAndDamp(int step) {
        const double diffusion = std::clamp(cfg_.cream.responsiveness * 0.26, 0.01,
                                            0.6);
        const double damping =
            std::clamp(cfg_.cream.viscosity * 0.18 + cfg_.cream.density * 0.03,
                       0.01, 0.65);
        const double resonance =
            0.014 * std::sin(step * cfg_.frother.frequency * 0.8); // mild lift

        for (int y = 0; y < cfg_.container.height; ++y) {
            for (int x = 0; x < cfg_.container.width; ++x) {
                double neighborSum = 0.0;
                int n = 0;

                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        if (ox == 0 && oy == 0) {
                            continue;
                        }
                        int nx = x + ox;
                        int ny = y + oy;

                        if (!inBounds(nx, ny)) {
                            if (cfg_.container.boundary == "reflective") {
                                nx = std::clamp(nx, 0, cfg_.container.width - 1);
                                ny = std::clamp(ny, 0, cfg_.container.height - 1);
                            } else {
                                continue;
                            }
                        }

                        neighborSum += grid_[idx(nx, ny)].energy;
                        ++n;
                    }
                }

                const double self = grid_[idx(x, y)].energy;
                const double neighborAvg = n > 0 ? neighborSum / n : 0.0;
                double updated = self + diffusion * (neighborAvg - self);
                updated *= (1.0 - damping);
                updated += std::max(0.0, resonance);

                scratch_[idx(x, y)].energy = std::max(0.0, updated);
                scratch_[idx(x, y)].foam = grid_[idx(x, y)].foam;
            }
        }

        grid_.swap(scratch_);
    }

    void updateFoam() {
        const double threshold = 0.15 + cfg_.cream.viscosity * 0.35;
        const double stabilityGain = 0.08 + cfg_.cream.viscosity * 0.08;
        const double decay = 0.03 + cfg_.cream.responsiveness * 0.04;

        for (int y = 0; y < cfg_.container.height; ++y) {
            for (int x = 0; x < cfg_.container.width; ++x) {
                double localAvg = grid_[idx(x, y)].energy;
                int n = 1;
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        if (ox == 0 && oy == 0) {
                            continue;
                        }
                        int nx = x + ox;
                        int ny = y + oy;
                        if (!inBounds(nx, ny)) {
                            continue;
                        }
                        localAvg += grid_[idx(nx, ny)].energy;
                        ++n;
                    }
                }
                localAvg /= n;

                auto &cell = grid_[idx(x, y)];
                if (localAvg > threshold) {
                    cell.foam = std::min(1.0, cell.foam + stabilityGain * localAvg);
                } else {
                    cell.foam = std::max(0.0, cell.foam - decay);
                }
            }
        }
    }

    Metrics measure() const {
        double totalEnergy = 0.0;
        double maxEnergy = 0.0;
        double foamMass = 0.0;

        std::vector<std::uint8_t> seen(grid_.size(), 0);
        int clusters = 0;

        for (std::size_t i = 0; i < grid_.size(); ++i) {
            totalEnergy += grid_[i].energy;
            maxEnergy = std::max(maxEnergy, grid_[i].energy);
            foamMass += grid_[i].foam;
        }

        const double clusterThreshold = 0.55;
        for (int y = 0; y < cfg_.container.height; ++y) {
            for (int x = 0; x < cfg_.container.width; ++x) {
                int start = idx(x, y);
                if (seen[start] || grid_[start].foam < clusterThreshold) {
                    continue;
                }
                ++clusters;

                std::vector<int> stack{start};
                seen[start] = 1;
                while (!stack.empty()) {
                    int current = stack.back();
                    stack.pop_back();
                    int cx = current % cfg_.container.width;
                    int cy = current / cfg_.container.width;

                    constexpr int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                    for (const auto &d : dirs) {
                        int nx = cx + d[0];
                        int ny = cy + d[1];
                        if (!inBounds(nx, ny)) {
                            continue;
                        }
                        int ni = idx(nx, ny);
                        if (!seen[ni] && grid_[ni].foam >= clusterThreshold) {
                            seen[ni] = 1;
                            stack.push_back(ni);
                        }
                    }
                }
            }
        }

        return Metrics{totalEnergy / grid_.size(), maxEnergy, foamMass, clusters};
    }

    std::string renderSnapshot(int step, const Metrics &m) const {
        std::ostringstream out;
        out << "\n--- Luscious Constraint | step " << step << " ---\n"
            << std::fixed << std::setprecision(3)
            << "mean_energy=" << m.meanEnergy << " max_energy=" << m.maxEnergy
            << " foam_mass=" << m.foamMass << " clusters=" << m.foamClusters
            << "\n";

        static const std::string ramp = " .:-=+*#%@";
        for (int y = 0; y < cfg_.container.height; ++y) {
            for (int x = 0; x < cfg_.container.width; ++x) {
                const auto &c = grid_[idx(x, y)];
                const double combined = std::clamp(0.55 * c.energy + 1.15 * c.foam,
                                                   0.0, 1.0);
                std::size_t bucket = static_cast<std::size_t>(
                    combined * static_cast<double>(ramp.size() - 1));
                out << ramp[bucket];
            }
            out << '\n';
        }
        return out.str();
    }
};

void applyArgs(int argc, char **argv, SimConfig &cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto expect = [&](const std::string &name) {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + name);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--width") {
            cfg.container.width = std::stoi(expect(arg));
        } else if (arg == "--height") {
            cfg.container.height = std::stoi(expect(arg));
        } else if (arg == "--steps") {
            cfg.totalSteps = std::stoi(expect(arg));
        } else if (arg == "--frother-intensity") {
            cfg.frother.intensity = std::stod(expect(arg));
        } else if (arg == "--frother-frequency") {
            cfg.frother.frequency = std::stod(expect(arg));
        } else if (arg == "--frother-duration") {
            cfg.frother.durationSteps = std::stoi(expect(arg));
        } else if (arg == "--viscosity") {
            cfg.cream.viscosity = std::stod(expect(arg));
        } else if (arg == "--density") {
            cfg.cream.density = std::stod(expect(arg));
        } else if (arg == "--responsiveness") {
            cfg.cream.responsiveness = std::stod(expect(arg));
        } else if (arg == "--boundary") {
            cfg.container.boundary = expect(arg);
        } else if (arg == "--csv") {
            cfg.csvPath = expect(arg);
        } else if (arg == "--seed") {
            cfg.seed = static_cast<std::uint32_t>(std::stoul(expect(arg)));
        } else if (arg == "--snapshot-every") {
            cfg.snapshotEvery = std::stoi(expect(arg));
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
}

int main(int argc, char **argv) {
    try {
        SimConfig cfg;
        applyArgs(argc, argv, cfg);
        LusciousConstraintSim sim(cfg);
        sim.run();
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n"
                  << "Usage: ./luscious_constraint [--width N --height N --steps N "
                     "--frother-intensity V --frother-frequency V "
                     "--frother-duration N --viscosity V --density V "
                     "--responsiveness V --boundary reflective|absorbent --csv "
                     "path --seed N --snapshot-every N]"
                  << std::endl;
        return 1;
    }
}
