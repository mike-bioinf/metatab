from resample_launchers.launchers import PolimorphicLauncher


def main():
    hard_launcher = PolimorphicLauncher(type_launcher="hard")
    hard_launcher.run()


if __name__ == "__main__":
    main()