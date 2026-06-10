import * as fs from "fs";
import * as path from "path";

interface ReadJsonFileOptions {
  readonly invalidJsonPrefix: string;
  readonly missingMessage?: string;
  readonly onMissing?: () => unknown;
  readonly readErrorPrefix: string;
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isFileNotFound(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    error.code === "ENOENT"
  );
}

export function readJsonFile(
  filePath: string,
  options: ReadJsonFileOptions,
): unknown {
  let fileContent: string;

  try {
    fileContent = fs.readFileSync(filePath, "utf8");
  } catch (error) {
    if (isFileNotFound(error)) {
      if (options.onMissing) {
        return options.onMissing();
      }
      if (options.missingMessage) {
        throw new Error(options.missingMessage);
      }
    }

    throw new Error(`${options.readErrorPrefix}: ${getErrorMessage(error)}`);
  }

  try {
    return JSON.parse(fileContent);
  } catch (error) {
    throw new Error(`${options.invalidJsonPrefix}: ${getErrorMessage(error)}`);
  }
}

export async function readJsonFileAsync(
  filePath: string,
  options: ReadJsonFileOptions,
): Promise<unknown> {
  let fileContent: string;

  try {
    fileContent = await fs.promises.readFile(filePath, "utf8");
  } catch (error) {
    if (isFileNotFound(error)) {
      if (options.onMissing) {
        return options.onMissing();
      }
      if (options.missingMessage) {
        throw new Error(options.missingMessage);
      }
    }

    throw new Error(`${options.readErrorPrefix}: ${getErrorMessage(error)}`);
  }

  try {
    return JSON.parse(fileContent);
  } catch (error) {
    throw new Error(`${options.invalidJsonPrefix}: ${getErrorMessage(error)}`);
  }
}

export function writeJsonFileAtomic(
  filePath: string,
  tempFilePrefix: string,
  config: unknown,
): void {
  const directory = path.dirname(filePath);
  fs.mkdirSync(directory, { recursive: true });

  const tempFilePath = path.join(
    directory,
    `.${tempFilePrefix}.${process.pid}.${Date.now()}.tmp`,
  );

  try {
    fs.writeFileSync(tempFilePath, `${JSON.stringify(config, null, 2)}\n`, {
      encoding: "utf8",
    });
    fs.renameSync(tempFilePath, filePath);
  } catch (error) {
    try {
      fs.rmSync(tempFilePath, { force: true });
    } catch {
      // Best-effort cleanup only.
    }
    throw error;
  }
}
