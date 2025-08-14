export default function KSSLogo({ className = "" }: { className?: string }) {
  return (
    <div className={`${className} bg-black dark:bg-white rounded-lg flex items-center justify-center`}>
      <span className="text-white dark:text-black font-bold text-xl">K</span>
    </div>
  );
}