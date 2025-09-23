from _paper.resample_launchers.launchers import PolimorphicLauncher


def main():
    soft_launcher = PolimorphicLauncher(type_launcher="soft")
    soft_launcher.run()


if __name__ == "__main__":
    main()