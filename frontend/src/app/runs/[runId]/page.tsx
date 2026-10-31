import { RunView } from "@/components/run/RunView";

/**
 * Route shell for the live run view.
 *
 * Deliberately class-free — see the note in `RunView` about Tailwind's scanner
 * and bracketed route directories.
 */
export default function Page() {
  return <RunView />;
}
